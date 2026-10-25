"""Behavioral tests using the real router/HTTP adapter and a fully offline provider."""

import json

import httpx
import pytest
from pydantic import SecretStr

from research_assist.agents.retrieval import RetrievalAgent
from research_assist.bootstrap import build_container
from research_assist.domain.coordination import make_task, selected_tasks, valid_requests
from research_assist.domain.models import (
    Claim,
    CoordinationDecision,
    FollowUpRequest,
    QuestionAspect,
    QuestionPlan,
    Requirement,
    ResearchRequest,
    RetrievalAction,
)
from research_assist.infrastructure.llm import DemoLanguageModel
from research_assist.infrastructure.search import NoReranker


class OfflineProvider:
    def __init__(
        self,
        action="retrieve",
        *,
        finish=False,
        invalid=False,
        contradict=False,
        complete=True,
    ):
        self.action, self.finish, self.invalid = action, finish, invalid
        self.contradict = contradict
        self.complete = complete
        self.calls = []
        self.checks = 0
        self.syntheses = 0

    def __call__(self, request):
        body = json.loads(request.content)
        task = body["response_format"]["json_schema"]["name"]
        payload = json.loads(body["messages"][1]["content"])
        self.calls.append((task, payload))
        if task == "question_plan":
            output = {
                "aspects": [{"text": text, "query": text} for text in payload["requirements"]]
            }
        elif task == "retrieval_action":
            output = {
                "action": "search",
                "queries": [
                    "targeted limitation evidence" if payload["requests"] else "initial method"
                ],
                "mode": "lexical" if payload["requests"] else "hybrid",
                "reason": "Search for the specific condition requested by the checker.",
            }
        elif task == "read_evidence":
            output = {
                "evidence": [
                    {
                        "chunk_id": u["chunk_id"],
                        "unit_id": u["unit_id"],
                        "quote": "",
                        "relevance": "Direct source evidence.",
                        "requirement_ids": ["rq_2" if payload["reading_requests"] else "rq_1"],
                    }
                    for u in payload["source_units"]
                ],
                "gaps": [],
            }
        elif task == "synthesize_claims":
            self.syntheses += 1
            ev = payload["evidence"]
            if self.syntheses == 1:
                claims = [
                    {
                        "text": ev[0]["quote"],
                        "evidence_ids": [ev[0]["id"]],
                        "requirement_ids": ["rq_1"],
                    }
                ]
                if self.action != "retrieve":
                    claims.append(
                        {
                            "text": "RAG always guarantees a fully correct answer.",
                            "evidence_ids": [ev[0]["id"]],
                            "requirement_ids": ["rq_2"],
                        }
                    )
            else:
                source = next((e for e in ev if "rq_2" in e["requirement_ids"]), ev[0])
                claims = [
                    {
                        "text": "The source limits the method to the conditions described.",
                        "evidence_ids": [source["id"]],
                        "requirement_ids": ["rq_2"],
                    }
                ]
            output = {"claims": claims, "gaps": []}
        elif task == "check_claims":
            self.checks += 1
            checks = []
            for c in payload["claims"]:
                verdict = "insufficient" if "always" in c["text"] else "supported"
                if self.action == "verify" and self.checks > 1 and self.complete:
                    verdict = "supported"
                if self.contradict and self.checks > 1 and "rq_1" in c["requirement_ids"]:
                    verdict = "contradicted"
                checks.append(
                    {
                        "claim_id": c["id"],
                        "verdict": verdict,
                        "evidence_ids": c["evidence_ids"],
                        "explanation": "Offline test verdict.",
                    }
                )
            supported = {c["claim_id"] for c in checks if c["verdict"] == "supported"}
            coverage = []
            for r in payload["requirements"]:
                ids = [
                    c["id"]
                    for c in payload["claims"]
                    if r["id"] in c["requirement_ids"] and c["id"] in supported
                ]
                coverage.append(
                    {
                        "requirement_id": r["id"],
                        "status": "covered" if ids else "missing",
                        "claim_ids": ids,
                        "explanation": "Offline test coverage.",
                    }
                )
            requests = []
            if any(r["requirement_id"] == "rq_2" and r["status"] != "covered" for r in coverage):
                bad = [c["id"] for c in payload["claims"] if "rq_2" in c["requirement_ids"]]
                requests = [
                    {
                        "action": self.action,
                        "requirement_ids": ["rq_2"],
                        "claim_ids": bad,
                        "need": "Find the specific limitation and its conditions.",
                        "chunk_ids": [payload["evidence"][0]["chunk_id"]]
                        if self.action == "read_context"
                        else [],
                    }
                ]
            output = {
                "checks": checks,
                "coverage": coverage,
                "gap_checks": [],
                "requests": requests,
            }
        elif task == "supervise":
            selected = payload["available_tasks"][0]
            output = {
                "action": "finish" if self.finish else selected["action"],
                "task_ids": []
                if self.finish
                else ["task_invented" if self.invalid else selected["id"]],
                "reason": "Resolve this concrete work request within the remaining budget.",
            }
        else:
            raise AssertionError(f"Unexpected task: {task}")
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )


class ScopedSearch:
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.calls, self.expansions = [], 0

    async def search(self, queries, document_ids, top_k, mode):
        self.calls.append((queries, document_ids, mode))
        target = self.second if "targeted limitation evidence" in queries else self.first
        assert target.document_id in document_ids
        return [target], []

    async def expand_context(self, chunks, document_ids):
        self.expansions += 1
        assert all(c.document_id in document_ids for c in chunks)
        return chunks


async def run_offline(
    settings, provider, *, max_calls=12, max_steps=4, max_rounds=2, mode="adaptive"
):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-test-key")
    settings.orchestration_mode = mode
    settings.llm_max_calls = max_calls
    settings.coordinator_max_steps = max_steps
    settings.llm_requests_per_minute = 1000
    container = build_container(settings, httpx.MockTransport(provider))
    for name, content in [
        ("method.txt", b"RAG uses retrieval evidence to ground generated answers."),
        ("limitations.txt", b"The method is limited to the conditions in this study."),
        ("excluded.txt", b"Excluded evidence must never enter the research scope."),
    ]:
        await container.documents.ingest(content, name)
    docs = {d.filename: d.id for d in container.repository.documents()}
    selected = [docs["method.txt"], docs["limitations.txt"]]
    from research_assist.domain.models import RetrievedChunk

    first, second = [
        RetrievedChunk(
            **container.repository.chunks([docs[name]])[0].model_dump(
                exclude={"vector", "vector_model"}
            ),
            score=1.0,
        )
        for name in ["method.txt", "limitations.txt"]
    ]
    search = ScopedSearch(first, second)
    container.runs.pipeline.retrieval.search = search
    try:
        record = container.runs.submit(
            ResearchRequest(
                question="Explain RAG evidence",
                requirements=["Method", "Limitations"],
                language="en",
                document_ids=selected,
                scope_mode="selected",
                max_rounds=max_rounds,
                use_cache=False,
            )
        )
        await container.runs.tasks[record.id]
        finished = container.repository.run(record.id)
        assert finished.status == "completed", finished.error
        return finished.result, search, selected
    finally:
        await container.close()


async def test_checker_request_drives_new_query_and_preserves_verified_claim(settings):
    provider = OfflineProvider()
    result, search, selected = await run_offline(settings, provider)
    assert result.stop_reason == "requirements_covered"
    assert result.orchestration_mode == "adaptive" and result.rounds == 2
    assert all(c.status == "covered" for c in result.coverage)
    assert len(result.reported_claim_ids) == 2
    assert {e.filename for e in result.evidence} == {"method.txt", "limitations.txt"}
    assert all(scope == selected for _, scope, _ in search.calls)
    assert any(
        queries == ["targeted limitation evidence"] and mode == "lexical"
        for queries, _, mode in search.calls
    )
    requests = [p for t, p in provider.calls if t == "retrieval_action"][-1]["requests"]
    assert requests[0]["need"] == "Find the specific limitation and its conditions."
    assert [p for t, p in provider.calls if t == "synthesize_claims"][-1]["requirements"] == [
        {"id": "rq_2", "text": "Limitations"}
    ]
    assert result.agent_tasks[0].status == "resolved"
    assert "supervise" not in [t for t, _ in provider.calls]
    assert "write_report" not in [t for t, _ in provider.calls]
    assert result.llm_calls == len(provider.calls)


@pytest.mark.parametrize("action", ["revise", "verify", "read_context"])
async def test_code_dispatches_repair_without_retrieving_again(settings, action):
    provider = OfflineProvider(action)
    result, search, _ = await run_offline(settings, provider)
    assert result.stop_reason == "requirements_covered"
    assert result.rounds == 1
    assert sum(t == "retrieval_action" for t, _ in provider.calls) == 1
    assert (
        result.agent_tasks[0].assigned_to
        == {
            "revise": "summarizer",
            "verify": "fact_checker",
            "read_context": "reader",
        }[action]
    )
    assert provider.syntheses == (1 if action == "verify" else 2)
    assert search.expansions == (2 if action == "read_context" else 1)
    assert all(t.status == "resolved" for t in result.agent_tasks)


async def test_budget_during_followup_keeps_last_verified_snapshot(settings):
    provider = OfflineProvider()
    result, _, _ = await run_offline(settings, provider, max_calls=8)
    assert result.stop_reason == "budget_exhausted"
    assert result.llm_calls == len(provider.calls) == 8
    assert len(result.reported_claim_ids) == 1
    assert result.claims[0].text == "RAG uses retrieval evidence to ground generated answers."
    assert all(e.filename == "method.txt" for e in result.evidence)
    assert result.coverage[0].status == "covered" and result.coverage[1].status == "missing"
    assert result.gap_details[0].reason == "budget_exhausted"
    assert result.agent_tasks[0].status == "unresolved"
    assert "write_report" not in [t for t, _ in provider.calls]


async def test_code_coordinator_ignores_retired_supervisor_decisions(settings):
    provider = OfflineProvider(finish=True, invalid=True)
    result, _, _ = await run_offline(settings, provider)
    assert result.stop_reason == "requirements_covered"
    assert len(result.reported_claim_ids) == 2
    assert "supervise" not in [t for t, _ in provider.calls]
    assert all(e.agent != "supervisor" for e in result.events)


async def test_new_evidence_rechecks_old_claims_instead_of_reusing_verdicts(settings):
    provider = OfflineProvider(contradict=True)
    result, _, _ = await run_offline(settings, provider)
    old = next(c for c in result.claims if "rq_1" in c.requirement_ids)
    assert old.id not in result.reported_claim_ids
    assert next(c for c in result.checks if c.claim_id == old.id).verdict == "contradicted"
    assert result.coverage[0].status != "covered"


async def test_repeating_task_on_unchanged_snapshot_stops_without_extra_verification(settings):
    provider = OfflineProvider("verify", complete=False)
    result, _, _ = await run_offline(settings, provider, max_steps=3)
    assert result.stop_reason == "no_progress"
    assert provider.checks == 2 and provider.syntheses == 1
    assert result.agent_tasks[0].attempts == 1
    assert result.coverage[1].status != "covered"


async def test_step_limit_keeps_incomplete_coverage(settings):
    provider = OfflineProvider("verify", complete=False)
    result, _, _ = await run_offline(settings, provider, max_steps=1)
    assert result.stop_reason == "step_limit"
    assert result.coverage[1].status == "missing"


async def test_code_coordinator_cannot_exceed_retrieval_round_limit(settings):
    provider = OfflineProvider()
    result, _, _ = await run_offline(settings, provider, max_rounds=1)
    assert result.stop_reason == "round_limit"
    assert result.rounds == 1 and result.coverage[1].status == "missing"
    assert "supervise" not in [t for t, _ in provider.calls]


async def test_fixed_academic_writer_renders_locally_after_call_budget(settings):
    settings.workflow_profile = "academic"
    provider = OfflineProvider()
    result, _, _ = await run_offline(
        settings,
        provider,
        mode="fixed",
        max_calls=4,
        max_rounds=1,
    )
    assert result.orchestration_mode == "fixed"
    assert result.stop_reason == "budget_exhausted"
    assert len(result.reported_claim_ids) == 1 and result.llm_calls == 4
    assert "write_report" not in [t for t, _ in provider.calls]


def test_work_requests_reject_unknown_ids_and_mismatched_dispatch(evidence):
    claim = Claim(
        id="cl_test", text=evidence.quote, evidence_ids=[evidence.id], requirement_ids=["rq_1"]
    )
    base = FollowUpRequest(
        action="retrieve", requirement_ids=["rq_1"], need="Find missing evidence."
    )
    invalid = [
        base.model_copy(update={"requirement_ids": ["rq_unknown"]}),
        base.model_copy(update={"claim_ids": ["cl_unknown"]}),
        base.model_copy(update={"chunk_ids": ["ch_unknown"]}),
        base.model_copy(update={"action": "read_context"}),
        base.model_copy(update={"action": "revise"}),
    ]
    assert (
        valid_requests(invalid, [Requirement(id="rq_1", text="Method")], [claim], [evidence]) == []
    )
    task = make_task(base)
    assert (
        selected_tasks(
            CoordinationDecision(
                action="verify", task_ids=[task.id], reason="Check the claim again."
            ),
            [task],
        )
        is None
    )
    assert (
        selected_tasks(
            CoordinationDecision(
                action="finish", task_ids=[task.id], reason="Stop with partial results."
            ),
            [task],
        )
        is None
    )


async def test_retrieval_observes_empty_result_and_changes_its_own_query(chunk):
    class Search:
        def __init__(self):
            self.calls = []

        async def search(self, queries, document_ids, top_k, mode):
            self.calls.extend(queries)
            return ([chunk] if queries == ["changed query"] else []), []

    class Decisions(DemoLanguageModel):
        def __init__(self):
            self.payloads = []

        async def generate(self, **kwargs):
            self.payloads.append(kwargs["payload"])
            return RetrievalAction(
                action="search",
                queries=["empty query" if len(self.payloads) == 1 else "changed query"],
                reason="Adapt to the observed empty search result.",
            )

    llm, search = Decisions(), Search()
    agent = RetrievalAgent(llm, search, NoReranker())
    _, chunks, _ = await agent.investigate(
        ResearchRequest(question="Explain RAG evidence", document_ids=[chunk.document_id]),
        QuestionPlan(aspects=[QuestionAspect(text="Method", query="method")]),
        [],
        None,
        set(),
        [],
    )
    assert chunks == [chunk]
    assert llm.payloads[1]["observations"][0]["new_chunks"] == 0
    assert "empty query" in search.calls and "changed query" in search.calls
