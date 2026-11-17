import asyncio
from typing import Literal

import pytest

from research_assist.bootstrap import build_container
from research_assist.config import Settings
from research_assist.domain.errors import CapacityError
from research_assist.domain.models import ResearchRequest, RunRecord
from research_assist.infrastructure.repository import Repository, now


@pytest.mark.parametrize(
    ("mode", "requirements"),
    [
        ("fixed", []),
        ("autonomous", []),
        ("autonomous", ["Compare methodology", "Compare sampling design"]),
    ],
)
async def test_demo_run_builds_comparison_from_frozen_document_titles(
    settings: Settings, mode: Literal["fixed", "autonomous"], requirements: list[str]
) -> None:
    container = build_container(settings)
    try:
        first, _ = await container.documents.ingest(
            b"The first empirical study describes methodology and sampling design in detail.",
            "first.txt",
            paper_title="Empirical paper A",
        )
        second, _ = await container.documents.ingest(
            b"The second empirical study explains methodology and sampling design using datasets.",
            "second.txt",
            paper_title="Empirical paper B",
        )
        missing, _ = await container.documents.ingest(
            b"A historical catalog of garden butterflies and their wing patterns.",
            "missing.txt",
            paper_title="Unobserved paper",
        )
        run = container.runs.submit(
            ResearchRequest(
                question="Compare empirical methodology and sampling design.",
                style="comparison",
                language="en",
                orchestration_mode=mode,
                scope_mode="selected",
                document_ids=[first.id, missing.id, second.id],
                requirements=requirements,
                max_rounds=1,
            )
        )
        container.repository.update_document_metadata(
            missing.model_copy(update={"title": "Changed after submission"})
        )
        await asyncio.wait_for(container.runs.tasks[run.id], 5)
        completed = container.repository.run(run.id)

        assert completed.status == "completed", completed.error
        assert completed.result is not None
        result = completed.result
        assert result.provider == "demo"
        assert "| Criterion | Empirical paper A | Unobserved paper | Empirical paper B |" in (
            result.markdown
        )
        assert " | Insufficient evidence | " in result.markdown
        assert "Changed after submission" not in result.markdown
        assert "The first empirical study" in result.markdown
        assert "The second empirical study" in result.markdown
        assert result.citation_audit.status == "passed"
        assert len(result.citation_evidence_ids) >= 2
    finally:
        await container.close()


async def test_cancel_running_and_queued_jobs(settings: Settings) -> None:
    container = build_container(settings)
    try:
        await container.documents.seed()
        gate = asyncio.Event()
        original = container.runs.pipeline.run

        async def slow(*args: object, **kwargs: object) -> object:
            await gate.wait()
            return await original(*args, **kwargs)

        container.runs.pipeline.run = slow
        container.runs.semaphore = asyncio.Semaphore(1)
        first = container.runs.submit(ResearchRequest(question="Explain RAG evidence"))
        second = container.runs.submit(ResearchRequest(question="Explain Self-RAG evidence"))
        await asyncio.sleep(0)
        assert container.repository.run(first.id).status == "running"
        assert container.repository.run(second.id).status == "queued"
        assert (await container.runs.cancel(second.id)).status == "cancelled"
        assert (await container.runs.cancel(first.id)).status == "cancelled"
        assert not container.runs.busy
    finally:
        await container.close()


async def test_queue_capacity_is_bounded(settings: Settings) -> None:
    container = build_container(settings)
    try:
        await container.documents.seed()
        settings.max_pending_runs = 1
        first = container.runs.submit(ResearchRequest(question="Explain RAG evidence"))
        with pytest.raises(CapacityError):
            container.runs.submit(ResearchRequest(question="Explain Self-RAG evidence"))
        await container.runs.cancel(first.id)
    finally:
        await container.close()


def test_restart_marks_interrupted_jobs_as_failed(settings: Settings) -> None:
    repository = Repository(settings.data_dir)
    request = ResearchRequest(question="Explain RAG evidence")
    for status in ("queued", "running", "completed"):
        repository.save_run(
            RunRecord(id=status, status=status, request=request, created_at=now(), updated_at=now())
        )
    repository.recover_interrupted()
    assert repository.run("queued").status == "failed"
    assert repository.run("running").status == "failed"
    assert repository.run("completed").status == "completed"
