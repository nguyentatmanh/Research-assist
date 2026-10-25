import json
from pathlib import Path

from fastapi.testclient import TestClient

from research_assist.api.app import create_app
from research_assist.config import Settings
from research_assist.domain.metadata import EVIDENCE_POLICY
from research_assist.domain.models import Document, RunRecord
from research_assist.experiments.system_benchmark import benchmark
from research_assist.infrastructure.repository import Repository
from tests.test_api import complete


def test_schema_adapter_reads_document_without_mutating_content(
    settings: Settings, document: Document
) -> None:
    repo = Repository(settings.data_dir)
    payload = document.model_dump(exclude={"warnings"})
    with repo.connection() as db:
        db.execute(
            "INSERT INTO documents VALUES (?,?,?,?)",
            (document.id, document.sha256, json.dumps(payload), "old.pdf"),
        )
        db.execute("PRAGMA user_version=2")
    upgraded = Repository(settings.data_dir)
    assert upgraded.document(document.id).warnings == []
    with upgraded.connection() as db:
        assert json.loads(db.execute("SELECT metadata FROM documents").fetchone()[0]) == payload
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_legacy_results_and_new_coverage_survive_restart(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.post("/api/documents/demo")
        run = client.post(
            "/api/runs",
            json={
                "question": "Explain RAG retrieval and evidence",
                "max_rounds": 1,
                "requirements": ["Explain retrieval", "Explain evidence"],
            },
        ).json()
        finished = complete(client, run["id"])
        assert finished["status"] == "completed"
        assert len(finished["result"]["coverage"]) == 2
        assert all(c["status"] == "unassessed" for c in finished["result"]["coverage"])
        assert len(client.get("/api/agents").json()["agents"]) == 5
        assert client.get("/api/diagnostics").json()["search_cache"]["builds"] == 1
        result = finished["result"]
        for key in [
            "coverage",
            "requirements",
            "reported_claim_ids",
            "stop_reason",
            "policy_version",
        ]:
            result.pop(key)
        finished["request"].pop("requirements")
        assert RunRecord.model_validate(finished).result.policy_version == "unclassified"
    with TestClient(create_app(settings)) as client:
        loaded = client.get(f"/api/runs/{run['id']}").json()
        assert loaded["result"]["policy_version"] == EVIDENCE_POLICY


async def test_system_benchmark_is_isolated_and_does_not_claim_quality() -> None:
    dataset = Path(__file__).parents[1] / "examples/benchmark/cases.jsonl"
    result = await benchmark(dataset, live=False, profile="balanced", max_rounds=1)
    assert result["summary"]["completed"] == 5
    assert result["summary"]["human_faithfulness"] is None
    assert all(row["result"]["llm_calls"] == 0 for row in result["cases"])


async def test_second_round_can_use_new_evidence_when_first_pool_is_full(evidence) -> None:
    from research_assist.agents.writer import WriterAgent
    from research_assist.application.pipeline import ResearchPipeline
    from research_assist.domain.models import (
        Claim,
        ClaimCheck,
        FactCheckOutput,
        ResearchRequest,
        RetrievalPlan,
    )
    from research_assist.infrastructure.llm import DemoLanguageModel

    first = [
        evidence.model_copy(
            update={
                "id": f"ev_{i}",
                "quote": f"Old source evidence number {i} describes retrieval.",
            }
        )
        for i in range(24)
    ]
    fresh = evidence.model_copy(
        update={"id": "ev_new", "quote": "New source evidence answers the missing question."}
    )

    class Retrieval:
        async def prepare(self, request):
            from research_assist.domain.planning import local_plan

            return local_plan(request)

        async def run(
            self, request, gaps, question_plan=None, pending_ids=None, seen_chunk_ids=None
        ):
            return RetrievalPlan(queries=[request.question]), [object()], []

    class Reader:
        count = 0

        async def run(self, question, chunks, requirements=None):
            self.count += 1
            return (first if self.count == 1 else [fresh]), [], []

    class Synthesis:
        async def run(self, question, pool, language, requirements):
            source = pool[0]
            return [Claim(id="cl_current", text=source.quote, evidence_ids=[source.id])], [], []

    class Checker:
        async def assess(self, question, claims, pool, requirements, gaps=None):
            claim = claims[0]
            verdict = "supported" if claim.evidence_ids == ["ev_new"] else "insufficient"
            return FactCheckOutput(
                checks=[
                    ClaimCheck(
                        claim_id=claim.id,
                        verdict=verdict,
                        explanation="Fixture judgement",
                        evidence_ids=claim.evidence_ids,
                    )
                ]
            )

    # Real chunk type is needed for provenance accumulation even with a stub retriever.
    from research_assist.domain.models import RetrievedChunk

    chunk = RetrievedChunk(
        id="ch_fixture",
        document_id=evidence.document_id,
        document_title=evidence.document_title,
        filename=evidence.filename,
        text=fresh.quote,
        page=1,
        section="fixture",
        ordinal=0,
        start=0,
        end=len(fresh.quote),
        score=0.1,
    )

    class ActualRetrieval(Retrieval):
        async def run(
            self, request, gaps, question_plan=None, pending_ids=None, seen_chunk_ids=None
        ):
            return RetrievalPlan(queries=[request.question]), [chunk], []

    llm = DemoLanguageModel()
    pipeline = ResearchPipeline(
        llm, ActualRetrieval(), Reader(), Synthesis(), Checker(), WriterAgent(llm), None
    )
    result = await pipeline.run(
        "run_fixture",
        ResearchRequest(
            question="Explain the missing research evidence", orchestration_mode="fixed"
        ),
        lambda event: None,
    )
    assert result.rounds == 2 and result.evidence[0].id == "ev_new"
    assert fresh.quote in result.markdown and len(result.evidence) == 24
