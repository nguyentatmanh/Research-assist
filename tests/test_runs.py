import asyncio

import pytest

from research_assist.bootstrap import build_container
from research_assist.config import Settings
from research_assist.domain.errors import CapacityError
from research_assist.domain.models import ResearchRequest, RunRecord
from research_assist.infrastructure.repository import Repository, now


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
