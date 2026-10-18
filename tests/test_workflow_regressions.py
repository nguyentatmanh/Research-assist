"""Behavioral regressions for scope, negative claims, question coverage and provenance."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.writer import WriterAgent
from research_assist.api.app import create_app
from research_assist.bootstrap import build_container
from research_assist.domain.coverage import reconcile_gaps, requirements_for
from research_assist.domain.models import (
    Claim,
    ClaimCheck,
    CoverageCheck,
    GapCheck,
    QuestionAspect,
    QuestionPlan,
    ResearchRequest,
    SourceSection,
)
from research_assist.domain.source_units import source_units
from research_assist.infrastructure.documents import chunk_sections
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.search import NoReranker
from tests.conftest import ScriptedLLM
from tests.test_api import complete


def test_empty_selected_scope_fails_closed():
    with pytest.raises(ValidationError):
        ResearchRequest(question="Explain the document", scope_mode="selected")


async def test_reindex_blocks_run_and_delete_until_vectors_are_committed(settings):
    import httpx

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        container = app.state.container
        doc, _ = await container.documents.ingest(
            b"RAG evidence retrieval methods support grounded scientific answers.", "a.txt"
        )
        entered, release = asyncio.Event(), asyncio.Event()

        class PausedEmbedder:
            fingerprint = "test-vectors"

            async def embed(self, texts, *, query=False):
                entered.set()
                await release.wait()
                return [[1.0, 0.0] for _ in texts]

        container.documents.embedder = PausedEmbedder()
        task = asyncio.create_task(container.documents.reindex())
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post("/api/runs", json={"question": "Explain RAG evidence"})
                assert response.status_code >= 400
                response = await client.delete(f"/api/documents/{doc.id}")
                assert response.status_code == 409
            assert container.documents.maintenance
        finally:
            release.set()
            await task
        assert not container.documents.maintenance
        assert container.repository.chunks([doc.id])[0].vector_model == "test-vectors"


async def test_all_scope_is_snapshotted_before_a_later_upload(settings):
    container = build_container(settings)
    try:
        original, _ = await container.documents.ingest(
            b"Original RAG retrieval evidence and methods.", "a.txt"
        )
        record = container.runs.submit(ResearchRequest(question="Explain RAG evidence"))
        later, _ = await container.documents.ingest(
            b"Later RAG retrieval evidence and methods.", "b.txt"
        )
        assert record.request.document_ids == [original.id]
        await asyncio.gather(*container.runs.tasks.values())
        result = container.repository.run(record.id).result
        assert result and result.document_ids == [original.id]
        assert all(c.document_id != later.id for c in result.retrieved)
    finally:
        await container.close()


def test_selected_scope_and_trace_survive_two_rounds(settings):
    with TestClient(create_app(settings)) as client:

        def upload(name, text):
            return client.post("/api/documents", files={"file": (name, text)}).json()["document"][
                "id"
            ]

        first = upload(
            "selected.txt", b"RAG retrieval uses scientific evidence to support an answer."
        )
        other = upload(
            "excluded.txt", b"RAG retrieval uses EXCLUDED_ONLY_SECRET evidence and experiments."
        )
        run = client.post(
            "/api/runs",
            json={
                "question": "Explain RAG retrieval and evidence",
                "scope_mode": "selected",
                "document_ids": [first],
                "max_rounds": 2,
            },
        ).json()
        finished = complete(client, run["id"])
        assert finished["status"] == "completed", finished["error"]
        result = finished["result"]
        assert result["rounds"] == 2
        assert all(c["document_id"] == first for c in result["retrieved"])
        assert "EXCLUDED_ONLY_SECRET" not in result["markdown"]
        trace = client.get(f"/api/runs/{run['id']}/trace").json()
        assert trace["document_ids"] == [first] and other not in trace["document_ids"]
        events = trace["events"]
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
        assert any("candidates" in e["details"] for e in events)
        assert any("coverage" in e["details"] for e in events)


@pytest.mark.parametrize(
    "review",
    [
        [],
        [
            GapCheck(
                gap_index=0, status="resolved", evidence_ids=["unknown"], explanation="Untrusted"
            )
        ],
    ],
)
def test_unverified_gap_blocks_complete_coverage(evidence, review):
    reqs = requirements_for(ResearchRequest(question="Explain coordination graphs"))
    coverage = [
        CoverageCheck(
            requirement_id="rq_1", status="covered", claim_ids=["c"], explanation="Complete"
        )
    ]
    checked, gaps = reconcile_gaps(
        reqs, coverage, ["The document contains no coordination graphs"], review, [evidence]
    )
    assert checked[0].status == "unassessed"
    assert gaps[0].scope == "retrieved_context"


def test_resolved_gap_does_not_create_false_absence(evidence):
    reqs = requirements_for(ResearchRequest(question="Explain coordination graphs"))
    coverage = [
        CoverageCheck(
            requirement_id="rq_1", status="covered", claim_ids=["c"], explanation="Answered"
        )
    ]
    review = [
        GapCheck(
            gap_index=0,
            status="resolved",
            evidence_ids=[evidence.id],
            requirement_ids=["rq_1"],
            explanation="The supplied quote addresses it",
        )
    ]
    checked, gaps = reconcile_gaps(reqs, coverage, ["missing graphs"], review, [evidence])
    assert checked[0].status == "covered" and not gaps


def test_out_of_scope_gap_does_not_override_covered_requirement(evidence):
    reqs = requirements_for(
        ResearchRequest(question="Which architecture was proposed and what did it replace?")
    )
    coverage = [
        CoverageCheck(
            requirement_id="rq_1", status="covered", claim_ids=["c"], explanation="Answered"
        )
    ]
    review = [
        GapCheck(
            gap_index=0,
            status="irrelevant",
            explanation="The question does not ask for positional encoding details.",
        )
    ]
    checked, gaps = reconcile_gaps(
        reqs,
        coverage,
        ["No positional encoding details in the retrieved quotes"],
        review,
        [evidence],
    )
    assert checked[0].status == "covered" and not gaps


async def test_writer_never_publishes_unchecked_absence(evidence):
    claim = Claim(id="c", text=evidence.quote, evidence_ids=[evidence.id])
    check = ClaimCheck(
        claim_id="c", verdict="supported", explanation="Exact", evidence_ids=[evidence.id]
    )
    report = await WriterAgent(DemoLanguageModel()).compose(
        ResearchRequest(question="Explain RAG evidence"),
        [claim],
        [check],
        [evidence],
        ["The document contains no coordination graphs."],
    )
    assert "contains no coordination" not in report.markdown
    assert not any("tiêu đề" in w for w in report.warnings)


async def test_reader_selects_units_without_generating_quotation(chunk):
    unit = source_units([chunk])[0]
    llm = ScriptedLLM(
        {
            "read_evidence": {
                "evidence": [
                    {
                        "chunk_id": chunk.id,
                        "unit_id": unit["unit_id"],
                        "quote": "",
                        "relevance": "Supports the requirement",
                        "requirement_ids": ["rq_1"],
                    },
                    {
                        "chunk_id": chunk.id,
                        "unit_id": "invented",
                        "quote": "",
                        "relevance": "Invalid selection",
                    },
                ],
                "gaps": [],
            }
        }
    )
    evidence, _, warnings = await ReaderAgent(llm).run("Explain RAG", [chunk], ["RAG"])
    assert len(evidence) == 1 and evidence[0].quote == unit["text"]
    assert evidence[0].start == chunk.start + unit["start"]
    assert evidence[0].requirement_ids == ["rq_1"]
    assert any("invalid_source_unit" in w for w in warnings)


async def test_reader_recovers_explicit_process_list_when_model_selects_adjacent_text(chunk):
    unrelated = chunk.model_copy(
        update={
            "id": "ch_context",
            "text": "Each clause has a representative word. Links must remain planar.",
            "start": 0,
            "end": 63,
        }
    )
    procedure = chunk.model_copy(
        update={
            "id": "ch_steps",
            "text": (
                "The analysis follows these steps:\n"
                "• Segment the sentence into clauses.\n"
                "• Parse each clause separately.\n"
                "• Combine the clause analyses."
            ),
            "start": 0,
            "end": 140,
        }
    )
    context_unit = source_units([unrelated])[0]
    llm = ScriptedLLM(
        {
            "read_evidence": {
                "evidence": [
                    {
                        "chunk_id": unrelated.id,
                        "unit_id": context_unit["unit_id"],
                        "quote": "",
                        "relevance": "Mentions clauses",
                    }
                ],
                "gaps": [],
            }
        }
    )
    evidence, _, warnings = await ReaderAgent(llm).run(
        "What are the steps in the analysis?", [unrelated, procedure], ["Steps"]
    )
    assert any(e.chunk_id == procedure.id and "Segment the sentence" in e.quote for e in evidence)
    assert any("bổ sung" in warning for warning in warnings)


async def test_reader_prefers_complete_process_list_over_truncated_window(chunk):
    partial = chunk.model_copy(
        update={
            "id": "ch_partial_steps",
            "text": (
                "The process has these steps:\n• Segment clauses.\n• Parse each clause.\n• Combine"
            ),
            "start": 0,
            "end": 78,
        }
    )
    complete = chunk.model_copy(
        update={
            "id": "ch_complete_steps",
            "text": (
                "The process has these steps:\n"
                "• Segment clauses.\n• Parse each clause.\n• Combine results."
            ),
            "start": 0,
            "end": 87,
        }
    )
    selected_unit = source_units([partial])[0]
    llm = ScriptedLLM(
        {
            "read_evidence": {
                "evidence": [
                    {
                        "chunk_id": partial.id,
                        "unit_id": selected_unit["unit_id"],
                        "quote": "",
                        "relevance": "A list of steps",
                    }
                ],
                "gaps": [],
            }
        }
    )
    evidence, _, _ = await ReaderAgent(llm).run(
        "What are the steps?", [partial, complete], ["Steps"]
    )
    assert any(e.chunk_id == complete.id and e.quote.endswith("Combine results.") for e in evidence)


async def test_reader_does_not_supplement_process_from_another_topic(chunk):
    unrelated = chunk.model_copy(
        update={
            "text": "The gardening process has these steps:\n• Plant seeds.\n• Water flowers.",
        }
    )
    llm = ScriptedLLM({"read_evidence": {"evidence": [], "gaps": []}})
    evidence, _, _ = await ReaderAgent(llm).run(
        "What are the Transformer training steps?", [unrelated], ["Transformer training"]
    )
    assert not evidence


def test_section_chunks_preserve_page_offsets(settings, document):
    text = "7.1 Protocols\n" + "Protocol bidding allocates tasks. " * 20
    text += "\n7.2 Coordination via Graphs\n" + "Variable elimination computes joint actions. " * 20
    source = SourceSection(text=text, page=21)
    chunks = chunk_sections([source], document, settings)
    assert any("7.2" in c.section for c in chunks)
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.page == 21
        assert not ("7.1 Protocols" in chunk.text and "7.2 Coordination" in chunk.text)


async def test_rebuild_uses_new_windows_and_rolls_back_on_failure(settings):
    container = build_container(settings)
    try:
        doc, _ = await container.documents.ingest(
            ("1 Methods\n" + "Retrieval evidence methods and measurements. " * 100).encode(),
            "paper.txt",
        )
        before = container.repository.chunks([doc.id])
        settings.chunk_chars = 500
        await container.documents.rebuild()
        rebuilt = container.repository.chunks([doc.id])
        assert len(rebuilt) > len(before)
        assert set(c.id for c in rebuilt).isdisjoint(c.id for c in before)
        assert container.repository.document(doc.id).index_version == "section-windows"

        class BrokenEmbedder:
            fingerprint = "broken"

            async def embed(self, texts, *, query=False):
                raise RuntimeError("Test failure")

        container.documents.embedder = BrokenEmbedder()
        with pytest.raises(RuntimeError):
            await container.documents.rebuild()
        assert [c.id for c in container.repository.chunks([doc.id])] == [c.id for c in rebuilt]
        assert not container.documents.maintenance
    finally:
        await container.close()


async def test_per_aspect_retrieval_retains_minority_topic(settings):
    container = build_container(settings)
    try:
        for name, content in [
            ("environment.txt", "Environment observability changes agent perception and sensors. "),
            ("graphs.txt", "Coordination graphs use variable elimination for joint actions. "),
            ("protocol.txt", "Protocol communication coordination bids agents tasks. "),
        ]:
            await container.documents.ingest((content * 15).encode(), name)
        plan = QuestionPlan(
            aspects=[
                QuestionAspect(text=x, query=x)
                for x in [
                    "environment perception",
                    "coordination graphs variable elimination",
                    "protocol bids",
                ]
            ]
        )
        request = ResearchRequest(
            question="Explain environment and protocols and graphs",
            top_k=4,
            retrieval_mode="lexical",
        )
        _, chunks, _ = await RetrievalAgent(
            DemoLanguageModel(), container.search, NoReranker()
        ).run(request, [], plan)
        assert {"environment.txt", "graphs.txt", "protocol.txt"}.issubset(
            c.filename for c in chunks
        )
    finally:
        await container.close()


async def test_live_events_persist_before_reader_finishes_and_cancel(settings):
    container = build_container(settings)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = container.runs.pipeline.reader.run

    async def blocked(question, chunks, requirements=None):
        entered.set()
        await release.wait()
        return await original(question, chunks, requirements)

    container.runs.pipeline.reader.run = blocked
    try:
        await container.documents.ingest(
            b"RAG retrieval and evidence support research methods.", "live.txt"
        )
        record = container.runs.submit(ResearchRequest(question="Explain RAG evidence"))
        await asyncio.wait_for(entered.wait(), timeout=3)
        running = container.repository.run(record.id)
        assert running.status == "running"
        assert running.events[-1].agent == "reader" and running.events[-1].status == "running"
        assert any(e.details.get("candidates") for e in running.events)
        cancelled = await container.runs.cancel(record.id)
        assert cancelled.status == "cancelled" and cancelled.events[-1].status == "cancelled"
    finally:
        release.set()
        await container.close()


async def test_rebuild_keeps_old_report_citations_readable(settings):
    container = build_container(settings)
    try:
        await container.documents.ingest(
            b"RAG retrieval uses evidence to support research answers.", "paper.txt"
        )
        record = container.runs.submit(
            ResearchRequest(question="Explain RAG evidence", max_rounds=1)
        )
        await asyncio.gather(*container.runs.tasks.values())
        before = container.repository.run(record.id)
        assert before.result and before.result.evidence
        await container.documents.rebuild()
        after = container.repository.run(record.id)
        assert after.result.model_dump() == before.result.model_dump()
        assert container.repository.blob(after.result.document_ids[0]).exists()
    finally:
        await container.close()
