"""Regression cases found while reviewing scope persistence and retrieval coverage."""

import re

import pytest
from fastapi.testclient import TestClient

from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.api.app import create_app
from research_assist.bootstrap import build_container
from research_assist.domain.models import QuestionAspect, QuestionPlan, ResearchRequest
from research_assist.domain.source_units import source_units
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.search import NoReranker
from tests.conftest import ScriptedLLM
from tests.test_api import complete


@pytest.mark.parametrize("scope_mode", ["all", "selected"])
def test_large_document_scope_survives_api_export_and_restart(settings, scope_mode):
    app = create_app(settings)
    with TestClient(app) as client:
        document_ids = []
        for index in range(51):
            response = client.post(
                "/api/documents",
                files={
                    "file": (
                        f"paper-{index}.txt",
                        f"RAG retrieval evidence supports research answers in study {index}.",
                    )
                },
            )
            assert response.status_code == 201
            document_ids.append(response.json()["document"]["id"])
        response = client.post(
            "/api/runs",
            json={
                "question": "Explain RAG retrieval evidence",
                "scope_mode": scope_mode,
                "document_ids": document_ids if scope_mode == "selected" else [],
                "max_rounds": 1,
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        record = complete(client, run_id)
        assert record["status"] == "completed", record
        assert set(record["request"]["document_ids"]) == set(document_ids)
        assert {doc["id"] for doc in record["scope_documents"]} == set(document_ids)
        assert set(record["result"]["document_ids"]) == set(document_ids)
        trace = client.get(f"/api/runs/{run_id}/trace")
        assert trace.status_code == 200
        assert set(trace.json()["document_ids"]) == set(document_ids)
        exported = client.get(f"/api/runs/{run_id}/export?format=json")
        assert exported.status_code == 200
        assert set(exported.json()["document_ids"]) == set(document_ids)
        client.post(
            "/api/documents",
            files={"file": ("later.txt", "Later RAG evidence must stay outside the saved scope.")},
        )
        # A pre-fix run interrupted after saving an expanded scope must also be recoverable.
        interrupted = app.state.container.repository.run(run_id).model_copy(
            update={"id": "run_interrupted_large_scope", "status": "queued", "result": None}
        )
        app.state.container.repository.save_run(interrupted)

    with TestClient(create_app(settings)) as client:
        restored = client.get(f"/api/runs/{run_id}")
        assert restored.status_code == 200
        assert restored.json()["result"] == record["result"]
        interrupted = client.get("/api/runs/run_interrupted_large_scope").json()
        assert interrupted["status"] == "failed"
        assert set(interrupted["request"]["document_ids"]) == set(document_ids)


@pytest.mark.parametrize("prefix_length", [0, 840, 1700])
@pytest.mark.parametrize("tail", ["Accuracy was 92%.", "No.", "Kết quả: 92%."])
def test_source_units_preserve_short_text_and_exact_coordinates(chunk, prefix_length, tail):
    text = ("A" * prefix_length + ". " if prefix_length else "") + tail
    chunk = chunk.model_copy(update={"text": text, "end": chunk.start + len(text)})
    units = source_units([chunk])
    assert any(tail in unit["text"] for unit in units)
    assert re.sub(r"\s+", "", "".join(unit["text"] for unit in units)) == re.sub(r"\s+", "", text)
    for unit in units:
        assert unit["text"] == text[unit["start"] : unit["end"]]
        assert len(unit["text"]) <= 869


async def test_reader_can_cite_short_tail_with_original_offsets(chunk):
    text = "A" * 840 + ". Accuracy was 92%."
    chunk = chunk.model_copy(update={"text": text, "end": chunk.start + len(text)})
    unit = next(u for u in source_units([chunk]) if "Accuracy was 92%." in u["text"])
    llm = ScriptedLLM(
        {
            "read_evidence": {
                "evidence": [
                    {
                        "chunk_id": chunk.id,
                        "unit_id": unit["unit_id"],
                        "quote": "",
                        "relevance": "Reports the measured accuracy",
                        "requirement_ids": ["rq_1"],
                    }
                ]
            }
        }
    )
    evidence, _, warnings = await ReaderAgent(llm).run(
        "What was the measured accuracy?", [chunk], ["Measured accuracy"]
    )
    assert not warnings
    assert len(evidence) == 1 and evidence[0].quote.endswith("Accuracy was 92%.")
    assert evidence[0].start == chunk.start + unit["start"]
    assert evidence[0].end == chunk.start + len(text)


@pytest.mark.parametrize("available", [0, 2, 20])
@pytest.mark.parametrize("candidate_limit", [4, 20])
async def test_overlapping_queries_fill_unique_slots_with_bounded_reranking(
    chunk, available, candidate_limit
):
    candidates = [chunk.model_copy(update={"id": f"ch_{i}"}) for i in range(available)]

    class Search:
        async def search(self, queries, document_ids, top_k, mode):
            assert document_ids == [chunk.document_id]
            assert mode == "lexical"
            return candidates[:top_k], []

    class Reranker:
        def __init__(self):
            self.seen = []

        async def rank(self, query, chunks):
            self.seen.extend(c.id for c in chunks)
            return list(reversed(chunks))

    reranker = Reranker()
    request = ResearchRequest(
        question="Explain all twelve research aspects",
        document_ids=[chunk.document_id],
        scope_mode="selected",
        top_k=12,
        retrieval_mode="lexical",
    )
    plan = QuestionPlan(
        aspects=[QuestionAspect(text=f"Aspect {i}", query=f"Query {i}") for i in range(12)]
    )
    trace, chunks, _ = await RetrievalAgent(
        DemoLanguageModel(), Search(), reranker, candidate_limit=candidate_limit
    ).run(request, [], plan)
    assert len(chunks) == min(available, candidate_limit, request.top_k)
    assert len({c.id for c in chunks}) == len(chunks)
    assert len(reranker.seen) == len(set(reranker.seen)) == min(available, candidate_limit)
    assert len(trace.trace) == len(trace.queries)
    assert sum(len(lane["before_rerank"]) for lane in trace.trace) == len(reranker.seen)
    if chunks:
        assert chunks[0].id == f"ch_{min(available, candidate_limit) - 1}"


async def test_real_search_fills_overlapping_aspects_without_leaking_scope(settings):
    container = build_container(settings)
    try:
        document_ids = []
        for index in range(20):
            document, _ = await container.documents.ingest(
                f"RAG retrieval evidence supports research answers in experiment {index}.".encode(),
                f"paper-{index}.txt",
            )
            document_ids.append(document.id)
        excluded, _ = await container.documents.ingest(
            b"RAG evidence RAG evidence PRIVATE_SCOPE_ONLY research findings.", "excluded.txt"
        )
        request = ResearchRequest(
            question="Explain RAG evidence across twelve aspects",
            document_ids=document_ids,
            scope_mode="selected",
            top_k=12,
            retrieval_mode="lexical",
        )
        plan = QuestionPlan(
            aspects=[
                QuestionAspect(text=f"Aspect {i}", query=f"RAG evidence aspect{i}")
                for i in range(12)
            ]
        )
        _, chunks, warnings = await RetrievalAgent(
            DemoLanguageModel(), container.search, NoReranker()
        ).run(request, [], plan)
        assert len(chunks) == request.top_k
        assert len({c.id for c in chunks}) == request.top_k
        assert all(c.document_id in document_ids and c.document_id != excluded.id for c in chunks)
        assert not warnings
    finally:
        await container.close()
