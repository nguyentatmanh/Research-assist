"""Acceptance tests use barriers inside search/read, never timing-based overlap."""

import asyncio
import json
import threading

import httpx
import pytest
from pydantic import SecretStr

from research_assist.agents.fact_checker import FactCheckerAgent
from research_assist.agents.reader import ReaderAgent
from research_assist.agents.retrieval import RetrievalAgent
from research_assist.agents.summarizer import SummarizerAgent
from research_assist.agents.writer import WriterAgent
from research_assist.application.citation_audit import audit_citations, verification_snapshot
from research_assist.application.parallel_research import ParallelResearchCoordinator, ScopedSources
from research_assist.application.pipeline import ResearchPipeline
from research_assist.application.research_state import ResearchState
from research_assist.application.task_execution import TaskExecutor
from research_assist.application.task_graph import TaskGraph
from research_assist.domain.errors import (
    ActionRejected,
    BudgetExceeded,
    ProviderUnavailable,
    TaskBudgetExceeded,
)
from research_assist.domain.models import (
    CoverageCheck,
    FollowUpRequest,
    QuestionAspect,
    QuestionPlan,
    ResearchRequest,
    ResearchTaskState,
    RetrievedChunk,
    RuntimeTask,
    TaskBudget,
    TaskRequest,
    TaskScope,
)
from research_assist.infrastructure.llm import (
    DemoLanguageModel,
    HTTPLanguageModel,
    current_task_budget,
    usage_scope,
)
from research_assist.infrastructure.search import NoReranker


def source(name, start=0, text=None):
    text = (
        text or f"Research {name} found reliable supporting evidence in the controlled experiment."
    )
    return RetrievedChunk(
        id="ch_" + name,
        document_id="doc_" + name,
        document_title=name,
        filename=name + ".txt",
        text=text,
        page=1,
        section="Results",
        ordinal=0,
        start=start,
        end=start + len(text),
        score=1.0,
    )


def request():
    return ResearchRequest(
        question="Compare research Alpha and Beta findings.",
        document_ids=["doc_Alpha", "doc_Beta"],
        requirements=["Research Alpha findings", "Research Beta findings"],
        orchestration_mode="autonomous",
        use_cache=False,
    )


def plan():
    return QuestionPlan(aspects=[QuestionAspect(text=t, query=t) for t in request().requirements])


class Search:
    def __init__(self, barrier=False, failures=(), shared=False):
        self.barrier, self.failures, self.shared = barrier, set(failures), shared
        self.entered = set()
        self.both = asyncio.Event()
        self.release_b = asyncio.Event()
        self.cancelled = set()
        self.read_a = asyncio.Event()
        self.calls = []

    async def search(self, queries, document_ids, top_k, mode):
        name = "Alpha" if "Alpha" in queries[0].splitlines()[0] else "Beta"
        self.calls.append(name)
        self.entered.add(name)
        if len(self.entered) == 2:
            self.both.set()
        if self.barrier:
            await asyncio.wait_for(self.both.wait(), 2)
        try:
            if self.barrier and name == "Beta":
                await self.release_b.wait()
            if name in self.failures:
                raise ProviderUnavailable("Injected local search failure.")
            return [
                source("Alpha", text="Research Alpha and Beta found reliable supporting evidence.")
                if self.shared
                else source(name)
            ], []
        except asyncio.CancelledError:
            self.cancelled.add(name)
            raise


class Reader(ReaderAgent):
    def __init__(self, llm, search, barrier=False):
        super().__init__(llm)
        self.search, self.barrier = search, barrier
        self.seen = []
        self.entered = set()
        self.both = asyncio.Event()

    async def run(self, question, chunks, requirements=None, requests=None):
        self.seen.append((requirements, [c.id for c in chunks]))
        name = "Alpha" if "Alpha" in question.splitlines()[0] else "Beta"
        self.entered.add(name)
        if name == "Alpha":
            self.search.read_a.set()
        if len(self.entered) == 2:
            self.both.set()
        if self.barrier:
            await asyncio.wait_for(self.both.wait(), 2)
        return await super().run(question, chunks, requirements, requests)


class Checker(FactCheckerAgent):
    async def assess(self, question, claims, evidence, requirements, gaps=None):
        result = await super().assess(question, claims, evidence, requirements, gaps)
        ids = {e.id: e for e in evidence}
        result.coverage = [
            CoverageCheck(
                requirement_id=r.id,
                status="covered"
                if any(
                    r.text.split()[1] in ids[eid].quote for c in claims for eid in c.evidence_ids
                )
                else "missing",
                claim_ids=[
                    c.id
                    for c in claims
                    if any(r.text.split()[1] in ids[eid].quote for eid in c.evidence_ids)
                ],
                explanation="Fake annotated requirement support.",
            )
            for r in requirements
        ]
        return result


def coordinator(search=None, reader_barrier=False, retries=1, **limits):
    search = search or Search()
    llm = DemoLanguageModel()
    reader = Reader(llm, search, reader_barrier)
    return ParallelResearchCoordinator(
        RetrievalAgent(llm, search, NoReranker()),
        reader,
        SummarizerAgent(llm),
        Checker(llm),
        WriterAgent(llm),
        retries=retries,
        **limits,
    ), reader


async def inline_step(agent, message, work, describe=None):
    return await work


async def run_team(team, state=None):
    state = state or ResearchState()
    with usage_scope():
        await team.run(request(), plan(), state, inline_step, lambda *a, **kw: None)
    return state


async def test_search_overlap_and_reader_a_starts_before_retrieval_b_finishes():
    search = Search(barrier=True)
    team, reader = coordinator(search)
    work = asyncio.create_task(run_team(team))
    try:
        await asyncio.wait_for(search.read_a.wait(), 3)
        assert search.entered == {"Alpha", "Beta"}
        assert not search.release_b.is_set()
        assert reader.seen == [(["Research Alpha findings"], ["ch_Alpha"])]
        search.release_b.set()
        state = await asyncio.wait_for(work, 3)
    finally:
        search.release_b.set()
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
    assert {e.document_id for e in state.evidence} == {"doc_Alpha", "doc_Beta"}
    assert all(t.status == "completed" for t in state.task_graph.values())
    assert state.citation_audit.accepted_claim_ids


async def test_reader_tools_overlap_and_never_receive_sibling_chunks():
    team, reader = coordinator(reader_barrier=True)
    await asyncio.wait_for(run_team(team), 3)
    assert reader.entered == {"Alpha", "Beta"}
    assert sorted(reader.seen) == [
        (["Research Alpha findings"], ["ch_Alpha"]),
        (["Research Beta findings"], ["ch_Beta"]),
    ]
    task = next(
        t
        for t in team.state.task_graph.values()
        if t.role == "reader" and t.requirement_ids == ["rq_1"]
    )
    with pytest.raises(ActionRejected, match="outside"):
        team.task_chunks(task.id, ["ch_Beta"])


async def test_failed_branch_is_retried_then_skipped_and_other_branch_is_reported():
    search = Search(failures={"Beta"})
    team, reader = coordinator(search)
    state = await run_team(team)
    beta = [t for t in state.task_graph.values() if t.requirement_ids == ["rq_2"]]
    assert [t.status for t in beta] == ["failed", "skipped"]
    assert beta[0].attempts == 2 and search.calls.count("Beta") == 2
    assert reader.entered == {"Alpha"} and not search.cancelled
    assert state.checks[0].verdict == "supported"
    assert state.citation_audit.accepted_claim_ids
    assert state.task_graph["job_5"].status == "completed"


def test_graph_terminal_join_waits_for_retry_and_success_edges_skip_failed_tasks():
    graph = TaskGraph({}, 10)
    a = graph.add(
        TaskRequest(role="retrieval", instruction="Find source evidence."), "orchestrator"
    )
    b = graph.add(
        TaskRequest(
            role="reader",
            instruction="Read source evidence.",
            depends_on=[a.id],
            dependency_conditions={a.id: "ON_SUCCESS"},
        ),
        "orchestrator",
    )
    join = graph.add(
        TaskRequest(
            role="summarizer",
            instruction="Merge available evidence.",
            depends_on=[b.id],
            dependency_conditions={b.id: "ON_TERMINAL"},
        ),
        "orchestrator",
    )
    a.status = "retry_waiting"
    graph.settle()
    assert join not in graph.ready() and b.status == "pending"
    a.status = "failed"
    graph.settle()
    assert b.status == "skipped" and graph.ready() == [join]


async def test_scope_expansion_cannot_change_anchor_or_introduce_unknown_documents():
    class Expansion(Search):
        async def expand_context(self, chunks, docs):
            return [source("Beta")]

    scope = TaskScope(task_id="a", requirement_ids=("rq_1",), allowed_document_ids=("doc_Alpha",))
    scoped = ScopedSources(Expansion(), scope, TaskBudget(), lambda: None)
    chunks, _ = await scoped.search(["Alpha"], ["doc_Alpha"], 8, "lexical")
    with pytest.raises(ActionRejected, match="anchor"):
        await scoped.expand_context(chunks, ["doc_Alpha"])
    with pytest.raises(ActionRejected, match="document"):
        await scoped.search(["Beta"], ["doc_Alpha"], 8, "lexical")


async def test_executor_rejects_dependency_changes_and_late_generation_without_refunds():
    executor = TaskExecutor()
    task = RuntimeTask(
        id="a", role="reader", instruction="Inspect scoped evidence.", status="running"
    )
    local = ResearchTaskState(
        task_id="a",
        scope=TaskScope(
            task_id="a", requirement_ids=("rq_1",), allowed_document_ids=("doc_Alpha",)
        ),
    )
    version, committed = ["old"], []

    async def work(ctx):
        local.budget.search_calls += 1
        version[0] = "new"
        return "result"

    with pytest.raises(ActionRejected):
        await executor.execute(task, local, lambda: version[0], work, committed.append)
    assert not committed and local.budget.search_calls == 1

    async def cancelled(ctx):
        task.generation += 1
        return "late"

    with pytest.raises(ActionRejected):
        await executor.execute(task, local, lambda: version[0], cancelled, committed.append)
    assert not committed


async def test_independent_reader_dependency_change_is_isolated():
    team, _ = coordinator()
    await run_team(team)
    a = team.state.task_graph["job_2"]
    before = team.dependency(a)
    team.chunks["job_4"]["ch_Beta"].text = "Changed only B."
    team.versions["doc_Beta"] = "new-version"
    assert team.dependency(a) == before
    team.chunks[a.id]["ch_Alpha"].text = "Changed A."
    assert team.dependency(a) != before


async def test_global_cancel_drains_running_search_and_prevents_followup_dispatch():
    search = Search(barrier=True)
    team, _ = coordinator(search)
    work = asyncio.create_task(run_team(team))
    await asyncio.wait_for(search.both.wait(), 2)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert "Beta" in search.cancelled
    assert not team.executor.active
    assert all(t.status in {"completed", "cancelled"} for t in team.state.task_graph.values())


async def test_shared_evidence_has_two_usage_links_and_immutable_locator():
    team, _ = coordinator(Search(shared=True))
    state = await run_team(team)
    assert len(state.evidence) == 1
    assert {u.requirement_id for u in state.evidence_usages} == {"rq_1", "rq_2"}
    assert len(state.evidence_usages) == 2
    assert state.evidence[0].document_version and state.evidence[0].requirement_ids == []


async def test_audit_rejects_quote_tampering_and_stale_verification():
    team, _ = coordinator()
    state = await run_team(team)
    ids = [c.id for c in state.claims]
    good = verification_snapshot(state)
    assert audit_citations(state, team.sources, team.versions, good, ids).accepted_claim_ids == ids
    state.evidence[0].quote = "Fabricated source content."
    assert audit_citations(state, team.sources, team.versions, good, ids).rejected_claim_ids == ids
    current = verification_snapshot(state)
    audit = audit_citations(state, team.sources, team.versions, current, ids)
    assert state.claims[0].id in audit.rejected_claim_ids


@pytest.mark.parametrize(
    "failures,status", [([], "complete"), (["Beta"], "partial"), (["Alpha", "Beta"], "abstained")]
)
async def test_pipeline_outputs_complete_partial_and_abstained_with_audit(failures, status):
    llm = DemoLanguageModel()
    search = Search(failures=failures)
    reader = Reader(llm, search)

    # Assign requirement ownership in the fake summarizer's fixture output.
    class Summary(SummarizerAgent):
        async def run(self, *args, **kwargs):
            claims, gaps, warnings = await super().run(*args, **kwargs)
            claims = [
                c.model_copy(update={"requirement_ids": ["rq_1" if "Alpha" in c.text else "rq_2"]})
                for c in claims
            ]
            return claims, gaps, warnings

    pipeline = ResearchPipeline(
        llm,
        RetrievalAgent(llm, search, NoReranker()),
        reader,
        Summary(llm),
        Checker(llm),
        WriterAgent(llm),
        None,
        orchestration_mode="autonomous",
    )
    checkpoints = []
    result = await asyncio.wait_for(
        pipeline.run("run_test", request(), lambda e: None, checkpoints.append), 3
    )
    assert result.report_status == status
    assert len(result.research_tasks) == 8
    assert result.citation_audit.status == ("not_applicable" if status == "abstained" else "passed")
    if status != "abstained":
        assert result.reported_claim_ids and all(
            saved.citation_audit.status == "passed" for saved in checkpoints
        )
    else:
        assert not result.reported_claim_ids and all(
            t.message for t in result.task_graph if t.status == "failed"
        )


@pytest.mark.parametrize("count,bad_reader", [(2, False), (12, False), (2, True)])
async def test_http_pipeline_uses_frozen_sources_and_persists_audited_result(
    settings, count, bad_reader
):
    from research_assist.bootstrap import build_container

    names = ["Alpha", "Beta"] if count == 2 else [f"Topic{i}" for i in range(count)]
    requirements = [f"Research {name} findings" for name in names]

    def provider(req):
        body = json.loads(req.content)
        payload = json.loads(body["messages"][1]["content"])
        task = body["response_format"]["json_schema"]["name"]
        if task == "question_plan":
            output = {"aspects": [{"text": text, "query": text} for text in requirements]}
        elif task == "read_evidence":
            unit = payload["source_units"][0]
            output = {
                "evidence": [
                    {
                        "chunk_id": unit["chunk_id"],
                        "unit_id": unit["unit_id"],
                        "quote": "",
                        "relevance": "Exact source.",
                    }
                ],
                "gaps": [],
            }
            if bad_reader and "Alpha" in payload["requirements"][0]:
                # Schema repair must not spend the reserved final verification calls.
                output = {"evidence": [{"chunk_id": "missing-required-fields"}]}
        elif task == "synthesize_claims":
            output = {
                "claims": [
                    {
                        "text": e["quote"],
                        "evidence_ids": [e["id"]],
                        "requirement_ids": e["requirement_ids"],
                    }
                    for e in payload["evidence"]
                ],
                "gaps": [],
            }
        else:
            assert task == "check_claims"
            output = {
                "checks": [
                    {
                        "claim_id": c["id"],
                        "verdict": "supported",
                        "evidence_ids": c["evidence_ids"],
                        "explanation": "Exact fixture source.",
                    }
                    for c in payload["claims"]
                ],
                "coverage": [
                    {
                        "requirement_id": r["id"],
                        "status": "covered",
                        "claim_ids": [
                            c["id"] for c in payload["claims"] if r["id"] in c["requirement_ids"]
                        ],
                        "explanation": "Annotated fixture coverage.",
                    }
                    for r in payload["requirements"]
                ],
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("test-only")
    settings.orchestration_mode = "autonomous"
    settings.llm_requests_per_minute = 1000
    settings.parallel_task_max_tokens = 60000
    settings.llm_max_calls = 5 if bad_reader else 12
    container = build_container(settings, transport=httpx.MockTransport(provider))
    try:
        docs = []
        for name in names:
            doc, _ = await container.documents.ingest(source(name).text.encode(), name + ".txt")
            docs.append(doc.id)
        req = request().model_copy(update={"document_ids": docs, "requirements": requirements})
        result = await container.runs.pipeline.run("run_http_mvp", req, lambda e: None)
        assert result.report_status == ("complete" if count == 2 and not bad_reader else "partial")
        assert result.citation_audit.status == "passed"
        assert len(result.evidence_usages) >= min(count, 9) - int(bad_reader)
        assert sum(t.budget.model_calls for t in result.research_tasks) == result.llm_calls - 1
        assert all(e.document_version for e in result.evidence)
        assert result.input_tokens == sum(a.input_tokens for a in result.agent_usage)
        assert result.llm_calls <= settings.llm_max_calls
        assert len([t for t in result.task_graph if t.role == "retrieval"]) == min(count, 9)
        if count == 12:
            assert result.stop_reason == "task_limit"
            assert any("rq_10" in warning for warning in result.warnings)
        if bad_reader:
            assert any(t.role == "reader" and t.status == "failed" for t in result.task_graph)
    finally:
        await container.close()


async def test_frozen_search_survives_source_deletion_and_rejects_foreign_scope(settings):
    from research_assist.application.documents import DocumentService
    from research_assist.infrastructure.embeddings import NoEmbeddings
    from research_assist.infrastructure.repository import Repository
    from research_assist.infrastructure.search import HybridSearch

    repo = Repository(settings.data_dir)
    doc, _ = await DocumentService(repo, NoEmbeddings(), settings).ingest(
        source("Alpha").text.encode(), "Alpha.txt"
    )
    live = HybridSearch(repo, NoEmbeddings(), settings)
    frozen = await live.freeze([doc.id])
    before, _ = await frozen.search(["Alpha"], [doc.id], 8, "lexical")
    repo.delete_document(doc.id)
    after, _ = await frozen.search(["Alpha"], [doc.id], 8, "lexical")
    assert before == after and after
    assert (await frozen.expand_context(after, [doc.id]))[0].text == before[0].text
    assert not (await live.search(["Alpha"], [doc.id], 8, "lexical"))[0]
    with pytest.raises(Exception, match="scope"):
        await frozen.search(["Alpha"], ["other"], 8, "lexical")


async def test_source_expansion_is_validated_and_recorded():
    class Expanded(Search):
        async def expand_context(self, chunks, docs):
            return [
                c.model_copy(
                    update={
                        "text": c.text + " Extra context.",
                        "end": c.end + len(" Extra context."),
                    }
                )
                for c in chunks
            ]

    team, _ = coordinator(Expanded())
    state = await run_team(team)
    for task in state.task_graph.values():
        if task.role == "retrieval":
            local = state.research_tasks[task.id]
            assert len(local.scope_history) == 2 and local.scope.allowed_chunk_ids
            assert local.scope_history[0].scope_version != local.scope_history[1].scope_version
            assert team.task_chunks(task.id)[0].text.endswith(" Extra context.")


async def test_same_quote_at_different_offsets_remains_separate_evidence():
    class Repeated(Search):
        async def search(self, queries, document_ids, top_k, mode):
            name = "Alpha" if "Alpha" in queries[0].splitlines()[0] else "Beta"
            if name == "Beta":
                return [], []
            a = source("Alpha")
            b = a.model_copy(update={"id": "ch_repeat", "start": 1000, "end": 1000 + len(a.text)})
            return [a, b], []

    team, _ = coordinator(Repeated())
    state = await run_team(team)
    assert len(state.evidence) == 2
    assert len({e.id for e in state.evidence}) == 2
    assert len({e.start for e in state.evidence}) == 2


async def test_concurrent_http_reserves_global_tokens_before_dispatch(settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("test-only")
    settings.llm_max_total_tokens = 2500
    entered, release = asyncio.Event(), asyncio.Event()

    async def provider(req):
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        model = HTTPLanguageModel(settings, client)
        with usage_scope() as usage:
            usage.strict_reservations = True
            first = asyncio.create_task(
                model._complete("x" * 1000, "q", {}, "test", output_token_limit=512)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                with pytest.raises(BudgetExceeded):
                    await model._complete("x" * 1000, "q", {}, "test", output_token_limit=512)
                assert usage.calls == 1
            finally:
                release.set()
                await first
            assert usage.reserved_tokens == 0


async def test_task_model_call_budget_is_local_and_does_not_consume_sibling_budget(settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("test-only")

    def provider(req):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        model = HTTPLanguageModel(settings, client)
        with usage_scope() as usage:
            usage.strict_reservations = True
            local = TaskBudget(max_model_calls=1)
            token = current_task_budget.set(local)
            try:
                await model._complete("q", "q", {}, "test", output_token_limit=512)
                with pytest.raises(TaskBudgetExceeded):
                    await model._complete("q", "q", {}, "test", output_token_limit=512)
                assert local.model_calls == 1 and local.reserved_tokens == 0
            finally:
                current_task_budget.reset(token)
            assert usage.calls == 1
            await model._complete("q", "q", {}, "sibling", output_token_limit=512)
            assert usage.calls == 2


async def test_cancelled_inference_keeps_physical_lock_until_worker_finishes(monkeypatch):
    import numpy as np

    from research_assist.infrastructure.embeddings import LocalEmbeddings
    from research_assist.infrastructure.local_runtime import INFERENCE_LOCK

    entered, release = threading.Event(), threading.Event()

    class Encoder:
        def encode(self, texts, **kwargs):
            entered.set()
            assert release.wait(3)
            return np.asarray([[1.0, 0.0]])

    embedding = LocalEmbeddings("test", 1)
    embedding._model = Encoder()
    task = asyncio.create_task(embedding.embed(["test"]))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not INFERENCE_LOCK.acquire(blocking=False)
    finally:
        release.set()
        # Wait for the actual worker, not its cancelled awaiting coroutine.
        await asyncio.to_thread(INFERENCE_LOCK.acquire)
        INFERENCE_LOCK.release()


async def test_cancelling_one_source_task_does_not_cancel_the_other_branch():
    class Cancellable(Search):
        async def search(self, queries, *args):
            if "Beta" in queries[0].splitlines()[0]:
                self.beta_task = asyncio.current_task()
                self.both.set()
                await asyncio.Event().wait()
            return await super().search(queries, *args)

    search = Cancellable()
    team, reader = coordinator(search)
    work = asyncio.create_task(run_team(team))
    await asyncio.wait_for(search.both.wait(), 2)
    search.beta_task.cancel()
    state = await asyncio.wait_for(work, 3)
    assert reader.entered == {"Alpha"}
    assert state.task_graph["job_3"].status == "cancelled"
    assert state.task_graph["job_4"].status == "skipped"
    assert state.citation_audit.accepted_claim_ids


def test_api_mvp_scope_and_audit_survive_repository_restart(settings):
    from fastapi.testclient import TestClient

    from research_assist.api.app import create_app
    from tests.test_api import complete

    settings.orchestration_mode = "autonomous"
    with TestClient(create_app(settings)) as client:
        client.post("/api/documents/demo")
        submitted = client.post(
            "/api/runs",
            json={
                "question": "Explain RAG retrieval and source evidence.",
                "requirements": ["Giải thích truy xuất", "Giải thích kiểm chứng"],
                "use_cache": False,
            },
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["id"]
        finished = complete(client, run_id)
        assert finished["status"] == "completed", finished
        result = finished["result"]
        assert len(result["research_tasks"]) == 8
        assert result["report_status"] in {"complete", "partial", "abstained"}
        assert result["citation_audit"]["status"] == "passed"
    with TestClient(create_app(settings)) as client:
        loaded = client.get(f"/api/runs/{run_id}").json()["result"]
        assert loaded["research_tasks"] == result["research_tasks"]
        assert loaded["citation_audit"] == result["citation_audit"]
        assert loaded["evidence_usages"] == result["evidence_usages"]


async def test_audit_removes_fabricated_claims_before_checkpoint_and_final_output():
    class BadReader(Reader):
        async def run(self, *args, **kwargs):
            evidence, gaps, warnings = await super().run(*args, **kwargs)
            return (
                [
                    e.model_copy(update={"quote": "Fabricated finding without a real source span."})
                    for e in evidence
                ],
                gaps,
                warnings,
            )

    llm, search = DemoLanguageModel(), Search()
    pipeline = ResearchPipeline(
        llm,
        RetrievalAgent(llm, search, NoReranker()),
        BadReader(llm, search),
        SummarizerAgent(llm),
        FactCheckerAgent(llm),
        WriterAgent(llm),
        None,
        orchestration_mode="autonomous",
    )
    checkpoints = []
    result = await pipeline.run("run_invalid_quote", request(), lambda e: None, checkpoints.append)
    assert result.report_status == "abstained" and not result.reported_claim_ids
    assert result.citation_audit.rejected_claim_ids or all(
        check.verdict != "supported" for check in result.checks
    )
    assert "Fabricated finding" not in result.markdown
    assert all(not saved.reported_claim_ids for saved in checkpoints)


@pytest.mark.parametrize("count,fail_last", [(3, False), (12, False), (3, True)])
async def test_n_way_branches_are_bounded_scoped_and_join_after_partial_failure(count, fail_last):
    class ManySearch(Search):
        def __init__(self):
            super().__init__()
            self.active = self.peak = 0
            self.pair = asyncio.Event()
            self.release = asyncio.Event()

        async def search(self, queries, document_ids, top_k, mode):
            name = queries[0].split()[1]
            self.calls.append(name)
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 2:
                self.pair.set()
            try:
                await self.release.wait()
                if fail_last and name == f"Topic{count - 1}":
                    raise ProviderUnavailable("One branch is unavailable.")
                return [source(name)], []
            finally:
                self.active -= 1

    req = request().model_copy(
        update={
            "requirements": [f"Research Topic{i} findings" for i in range(count)],
            "document_ids": [f"doc_Topic{i}" for i in range(count)],
        }
    )
    query_plan = QuestionPlan(aspects=[QuestionAspect(text=t, query=t) for t in req.requirements])
    search = ManySearch()
    team, reader = coordinator(search, retries=0, parallelism=2)
    state = ResearchState()

    async def run():
        with usage_scope():
            await team.run(req, query_plan, state, inline_step, lambda *a, **kw: None)

    work = asyncio.create_task(run())
    try:
        await asyncio.wait_for(search.pair.wait(), 3)
        assert len(search.calls) == 2
        search.release.set()
        await asyncio.wait_for(work, 3)
    finally:
        search.release.set()
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
    assert search.peak == 2 and team.search_calls == count
    assert len(state.task_graph) == 2 * count + 4
    assert len(state.evidence) == count - int(fail_last)
    assert all(ids == ["ch_" + texts[0].split()[1]] for texts, ids in reader.seen)
    summary = next(t for t in state.task_graph.values() if t.role == "summarizer")
    assert len(summary.depends_on) == count
    assert set(summary.dependency_conditions.values()) == {"ON_TERMINAL"}
    assert summary.requirement_ids == [f"rq_{i + 1}" for i in range(count)]
    assert len(state.citation_audit.accepted_claim_ids) == count - int(fail_last)
    if fail_last:
        assert state.coverage[-1].status == "missing"
        assert [
            t.status for t in state.task_graph.values() if t.requirement_ids == [f"rq_{count}"]
        ] == ["failed", "skipped"]


@pytest.mark.parametrize("limit", [1, 8, 32])
async def test_inferred_requirements_use_n_way_and_reserve_verification_budget(limit):
    class PlannedRetrieval(RetrievalAgent):
        async def prepare(self, req):
            return QuestionPlan(
                aspects=[
                    QuestionAspect(text=t, query=t)
                    for t in [*request().requirements, "Research Gamma findings"]
                ]
            )

    llm, search = DemoLanguageModel(), Search()
    pipeline = ResearchPipeline(
        llm,
        PlannedRetrieval(llm, search, NoReranker()),
        Reader(llm, search),
        SummarizerAgent(llm),
        Checker(llm),
        WriterAgent(llm),
        None,
        autonomous_max_tasks=limit,
        autonomous_max_steps=limit,
    )
    result = await pipeline.run(
        "run_inferred", request().model_copy(update={"requirements": []}), lambda e: None
    )
    assert len(result.requirements) == 3
    assert len(result.task_graph) <= limit
    if limit == 1:
        assert result.stop_reason == "budget_exhausted" and not search.calls
        assert result.report_status == "abstained"
    else:
        # The third initial branch has no supporting source, so with enough
        # slots it gets one more search before the result is reported.
        expected = 2 if limit == 8 else 4
        assert len([t for t in result.task_graph if t.role == "retrieval"]) == expected
        assert result.citation_audit.status == "passed"
        assert all(t.status == "completed" for t in result.task_graph if t.role == "writer")
        if limit == 8:
            assert result.stop_reason == "task_limit"
            assert result.coverage[-1].status != "covered"
            assert any("rq_3" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "outcome", ["repaired", "provider_error", "budget", "task_limit", "round_limit"]
)
async def test_critic_repair_is_bounded_and_keeps_previous_verified_snapshot(outcome):
    team, _ = coordinator(max_tasks=8 if outcome == "task_limit" else 32)
    llm = DemoLanguageModel()
    calls = []

    class Summary(SummarizerAgent):
        async def run(self, question, evidence, language, requirements=None, requests=None):
            claims, gaps, warnings = await super().run(
                question, evidence, language, requirements, requests
            )
            for claim in claims:
                claim.requirement_ids = ["rq_1" if "Alpha" in claim.text else "rq_2"]
            if not requests:
                claims[0] = claims[0].model_copy(update={"text": "Unsupported initial claim."})
            else:
                assert requests[0].claim_ids == [claims[0].id]
                calls.append("revise")
                if outcome in {"provider_error", "budget"}:
                    claims[0] = claims[0].model_copy(update={"text": "UNVERIFIED REPAIR DRAFT"})
            return claims, gaps, warnings

    class Critic(FactCheckerAgent):
        async def assess(self, question, claims, evidence, requirements, gaps=None):
            if calls:
                assert team.state.claims[0].text == "Unsupported initial claim."
                assert any(check.verdict == "supported" for check in team.state.checks)
                if outcome == "provider_error":
                    raise ProviderUnavailable("Repair checker unavailable.")
                if outcome == "budget":
                    raise BudgetExceeded("Repair exhausted the run budget.")
            result = await super().assess(question, claims, evidence, requirements, gaps)
            supported = {c.claim_id for c in result.checks if c.verdict == "supported"}
            result.coverage = [
                CoverageCheck(
                    requirement_id=r.id,
                    status="covered" if ids else "missing",
                    claim_ids=ids,
                    explanation="Fixture verifies only exact supported source claims.",
                )
                for r in requirements
                for ids in [
                    [c.id for c in claims if c.id in supported and r.id in c.requirement_ids]
                ]
            ]
            if not calls:
                result.requests = [
                    FollowUpRequest(
                        action="revise",
                        requirement_ids=claims[0].requirement_ids,
                        claim_ids=[claims[0].id],
                        need="Replace the unsupported statement with evidence.",
                    )
                ]
            return result

    team.summarizer, team.checker = Summary(llm), Critic(llm)
    state = ResearchState()
    req = request().model_copy(update={"max_rounds": 1 if outcome == "round_limit" else 2})
    with usage_scope():
        if outcome == "budget":
            with pytest.raises(BudgetExceeded):
                await team.run(req, plan(), state, inline_step, lambda *a, **kw: None)
        else:
            await team.run(req, plan(), state, inline_step, lambda *a, **kw: None)
    team.audit_now()
    assert len([t for t in state.task_graph.values() if t.role == "summarizer"]) <= 2
    assert all("UNVERIFIED" not in claim.text for claim in state.claims)
    if outcome == "repaired":
        assert len(state.citation_audit.accepted_claim_ids) == 2
        assert state.stop_reason == "requirements_covered"
    else:
        assert len(state.citation_audit.accepted_claim_ids) == 1
        assert state.claims[0].text == "Unsupported initial claim."
    if outcome in {"task_limit", "round_limit"}:
        assert not calls and state.rounds == 1


@pytest.mark.parametrize(
    "outcome",
    [
        "retrieved",
        "read_context",
        "multi_context",
        "implicit",
        "no_new",
        "failed",
        "outside",
        "context_outside",
        "section_outside",
    ],
)
async def test_focused_search_repair_stages_new_evidence_until_checked(outcome):
    class FocusedSearch(Search):
        async def read_section(self, seeds, document_ids):
            branch = seeds[0].document_title
            assert branch == "Alpha" or outcome == "multi_context"
            assert [c.id for c in seeds] == [f"ch_{branch}"]
            assert document_ids == request().document_ids
            self.calls.append(branch)
            items = [
                *seeds,
                source(
                    branch,
                    start=200,
                    text=f"Research {branch} reported missing robustness details across datasets.",
                ).model_copy(update={"id": f"ch_{branch}_detail"}),
            ]
            if outcome == "section_outside":
                items[-1].section = "Unrelated section"
            return items

        async def search(self, queries, document_ids, top_k, mode):
            result, notes = await super().search(queries, document_ids, top_k, mode)
            if (
                "missing robustness details" in queries[0] and outcome != "no_new"
            ) or "unanswered aspect: Research Alpha findings" in queries[0]:
                extra = source(
                    "Alpha",
                    start=200,
                    text="Research Alpha reported missing robustness details across datasets.",
                ).model_copy(update={"id": "ch_Alpha_detail"})
                if outcome == "outside":
                    extra.document_id = "doc_not_selected"
                return [*result, extra], notes
            return result, notes

    search = FocusedSearch()
    team, _ = coordinator(search, retries=0)
    assessments = []
    reader_requests = []

    class FocusedReader(Reader):
        async def run(self, question, chunks, requirements=None, requests=None):
            if requests:
                reader_requests.extend(requests)
                branch = requirements[0].split()[1]
                assert branch == "Alpha" or outcome == "multi_context"
                assert {chunk.document_id for chunk in chunks} <= {f"doc_{branch}"}
            return await super().run(question, chunks, requirements, requests)

    class Critic(Checker):
        async def assess(self, question, claims, evidence, requirements, gaps=None):
            if assessments:
                assert len(team.state.evidence) == 2
                assert verification_snapshot(team.state) == team.approved_snapshot
                assert len(evidence) == (4 if outcome == "multi_context" else 3)
                if outcome == "failed":
                    raise ProviderUnavailable("Repair verification failed.")
            result = await super().assess(question, claims, evidence, requirements, gaps)
            assessments.append(len(evidence))
            if len(assessments) == 1 and outcome == "implicit":
                result.coverage[0] = CoverageCheck(
                    requirement_id="rq_1",
                    status="missing",
                    explanation="Alpha's robustness details remain unanswered.",
                )
            elif len(assessments) == 1:
                result.requests = [
                    FollowUpRequest(
                        action="read_context"
                        if outcome
                        in {"read_context", "multi_context", "context_outside", "section_outside"}
                        else "retrieve",
                        requirement_ids=["rq_1", "rq_2"]
                        if outcome == "multi_context"
                        else ["rq_1"],
                        chunk_ids=["ch_Alpha", "ch_Beta"]
                        if outcome == "multi_context"
                        else ["ch_Beta" if outcome == "context_outside" else "ch_Alpha"],
                        need="Find missing robustness details for the Alpha study.",
                    )
                ]
            return result

    team.checker = Critic(DemoLanguageModel())
    team.reader = FocusedReader(DemoLanguageModel(), search)
    state = await run_team(team)
    team.audit_now()
    if outcome == "multi_context":
        assert len(search.calls) == 4 and len(state.task_graph) == 14
        assert sorted((r.requirement_ids, r.chunk_ids) for r in reader_requests) == [
            (["rq_1"], ["ch_Alpha"]),
            (["rq_2"], ["ch_Beta"]),
        ]
        assert assessments == [2, 4]
        assert len(state.evidence) == len(state.citation_audit.accepted_claim_ids) == 4
        return
    assert len(search.calls) == (2 if outcome == "context_outside" else 3)
    assert search.calls.count("Beta") == 1
    assert len(state.task_graph) == 12 and state.rounds == 2
    if outcome in {"retrieved", "read_context", "implicit"}:
        assert len(reader_requests) == 1
        assert reader_requests[0].requirement_ids == ["rq_1"]
        repair = [t for t in state.task_graph.values() if t.id in team.repair_readers]
        assert repair[0].requested_by == (
            "orchestrator" if outcome == "implicit" else "fact_checker"
        )
        assert assessments == [2, 3]
        assert len(state.evidence) == 3 and len(state.citation_audit.accepted_claim_ids) == 3
    else:
        assert len(state.evidence) == 2 and len(state.citation_audit.accepted_claim_ids) == 2
        assert all("missing robustness" not in c.text for c in state.claims)
    assert all(ev.document_id in request().document_ids for ev in state.evidence)
