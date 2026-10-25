"""Offline HTTP integration: real runtime decisions, peer dependencies and factual guards."""

import json
from collections import Counter

import httpx
import pytest
from pydantic import SecretStr

from research_assist.application.autonomous import merge_source_candidates
from research_assist.bootstrap import build_container
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import ResearchRequest, RetrievedChunk


def decision(action, **kwargs):
    return {
        "action": action,
        "reason": "Complete the document research task using supplied artifacts.",
        **kwargs,
    }


class AutonomousProvider:
    def __init__(
        self,
        *,
        writer_driven=False,
        budget_after_revision=False,
        invalid_writer=False,
        stale_writer=False,
        block_retrieval=False,
        padded_arguments=False,
        search_to_limit=False,
    ):
        self.calls, self.steps = [], Counter()
        self.writer_driven = writer_driven
        self.budget_after_revision = budget_after_revision
        self.invalid_writer = invalid_writer
        self.stale_writer = stale_writer
        self.block_retrieval = block_retrieval
        self.padded_arguments = padded_arguments
        self.search_to_limit = search_to_limit
        self.revision_scheduled = False
        self.summary_jobs = set()
        self.original_claim = None

    def __call__(self, request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        payload = json.loads(body["messages"][1]["content"])
        self.calls.append((name, payload))
        if name == "question_plan":
            output = {"aspects": [{"text": "RAG evidence", "query": "RAG evidence"}]}
            usage = 20
        else:
            task_id, context = payload["task_id"], payload["context"]
            self.steps[task_id] += 1
            count = self.steps[task_id]
            artifacts = context["artifacts"]
            usage = 20
            if name == "retrieval_step":
                if self.block_retrieval:
                    output = decision("blocked")
                elif count == 1 or self.search_to_limit:
                    output = decision(
                        "use_tool",
                        tool="search",
                        arguments={"queries": ["RAG evidence"], "search_mode": "lexical"},
                    )
                else:
                    chunks = payload["last_observation"]["chunks"]
                    output = decision("finish", result={"chunk_ids": [c["id"] for c in chunks]})
            elif name == "reader_step":
                units = context["source_units"]
                if count == 1:
                    output = decision(
                        "use_tool",
                        tool="read_section",
                        arguments={"chunk_ids": [units[0]["chunk_id"]]},
                    )
                else:
                    units = payload["last_observation"].get("source_units", units)
                    output = decision(
                        "finish",
                        result={
                            "evidence": [
                                {
                                    "chunk_id": u["chunk_id"],
                                    "unit_id": u["unit_id"],
                                    "quote": "",
                                    "relevance": "Exact relevant source text.",
                                    "requirement_ids": ["rq_1"],
                                }
                                for u in units[:2]
                            ],
                            "gaps": [],
                        },
                    )
            elif name == "summarizer_step":
                self.summary_jobs.add(task_id)
                if count == 1:
                    output = decision("use_tool", tool="inspect_evidence")
                else:
                    source = context["evidence"][0]
                    text = (
                        source["quote"]
                        if len(self.summary_jobs) == 1
                        else "This proposed revision has not been checked yet."
                    )
                    if self.original_claim is None:
                        self.original_claim = text
                    output = decision(
                        "finish",
                        result={
                            "claims": [
                                {
                                    "text": text,
                                    "evidence_ids": [source["id"]],
                                    "requirement_ids": ["rq_1"],
                                }
                            ],
                            "gaps": [],
                        },
                    )
                    if self.budget_after_revision and len(self.summary_jobs) > 1:
                        usage = 1000
            elif name == "fact_checker_step":
                if count == 1:
                    output = decision("use_tool", tool="check_numbers")
                else:
                    claims = context["claims"]
                    output = decision(
                        "finish",
                        result={
                            "checks": [
                                {
                                    "claim_id": c["id"],
                                    "verdict": "supported",
                                    "evidence_ids": c["evidence_ids"],
                                    "explanation": "Offline source-backed verdict.",
                                }
                                for c in claims
                            ],
                            "coverage": [
                                {
                                    "requirement_id": "rq_1",
                                    "status": "covered",
                                    "claim_ids": [c["id"] for c in claims],
                                    "explanation": "Offline complete coverage.",
                                }
                            ],
                            "gap_checks": [],
                            "requests": [],
                        },
                    )
            elif name == "writer_step":
                role = None
                if (
                    self.writer_driven or self.budget_after_revision or self.stale_writer
                ) and not self.revision_scheduled:
                    role = "summarizer"
                    self.revision_scheduled = True
                elif self.writer_driven and (
                    artifacts["checked_revision"] != artifacts["artifact_revision"]
                ):
                    role = "fact_checker"
                if (
                    self.stale_writer
                    and role is None
                    and artifacts["artifact_revision"] != artifacts["checked_revision"]
                ):
                    if payload["last_observation"].get("error"):
                        output = decision("blocked")
                    else:
                        output = decision(
                            "finish",
                            result={
                                "title": "Research notes",
                                "sections": [
                                    {
                                        "heading": "Evidence",
                                        "blocks": [
                                            {"claim_id": c["id"]}
                                            for c in context["verified_claims"]
                                        ],
                                    },
                                ],
                            },
                        )
                elif role:
                    output = decision(
                        "request_task",
                        arguments={
                            "task": {
                                "role": role,
                                "instruction": "Obtain the prerequisite for a verified report.",
                            }
                        },
                    )
                elif self.invalid_writer and count == 1:
                    output = decision(
                        "finish",
                        result={
                            "title": "Unchecked text",
                            "sections": [
                                {"heading": "Evidence", "blocks": [{"claim_id": "cl_invented"}]},
                            ],
                        },
                    )
                elif not self.writer_driven and count == 1:
                    output = decision("use_tool", tool="inspect_claims")
                else:
                    output = decision(
                        "finish",
                        result={
                            "title": "Research notes",
                            "sections": [
                                {
                                    "heading": "Evidence",
                                    "blocks": [
                                        {"claim_id": c["id"]} for c in context["verified_claims"]
                                    ],
                                },
                            ],
                        },
                    )
            else:
                raise AssertionError(f"Unexpected task: {name}")
        if self.padded_arguments and name.endswith("_step") and output.get("tool") != "search":
            output["arguments"] = output.get("arguments") or {}
            output["arguments"]["search_mode"] = "lexical"
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": usage, "completion_tokens": 5},
            },
        )


async def run_autonomous(
    settings, provider, *, max_calls=12, budget=60000, content=None, max_dispatches=16
):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-only-key")
    settings.orchestration_mode = "autonomous"
    settings.llm_requests_per_minute = 1000
    settings.llm_max_calls, settings.llm_max_total_tokens = max_calls, budget
    settings.autonomous_max_steps = max_dispatches
    container = build_container(settings, httpx.MockTransport(provider))
    try:
        doc, _ = await container.documents.ingest(
            content
            or (
                b"RAG uses retrieved source evidence to support generated answers. "
                b"The source describes limitations and conditions for applying this method."
            ),
            "paper.txt",
        )
        excluded, _ = await container.documents.ingest(
            b"RAG evidence EXCLUDED_SOURCE_ONLY should not enter the selected scope.",
            "excluded.txt",
        )
        record = container.runs.submit(
            ResearchRequest(
                question="Explain RAG evidence",
                language="en",
                document_ids=[doc.id],
                scope_mode="selected",
                use_cache=False,
            )
        )
        await container.runs.tasks[record.id]
        result = container.repository.run(record.id)
        assert result.status == "completed", result.error
        assert all(c.document_id != excluded.id for c in result.result.retrieved)
        return result.result
    finally:
        await container.close()


async def test_all_roles_use_private_runtime_loops_with_real_http_routing(settings):
    provider = AutonomousProvider()
    result = await run_autonomous(settings, provider)
    assert (
        result.orchestration_mode == "autonomous" and result.stop_reason == "requirements_covered"
    )
    assert len(result.reported_claim_ids) == 1
    assert {s.role for s in result.agent_sessions} == {
        "retrieval",
        "reader",
        "summarizer",
        "fact_checker",
        "writer",
    }
    assert all(s.status == "completed" for s in result.agent_sessions)
    assert all(s.steps >= (1 if s.role == "retrieval" else 2) for s in result.agent_sessions)
    assert len(result.task_graph) == 5 and all(t.status == "completed" for t in result.task_graph)
    assert result.llm_calls == len(provider.calls) <= 12
    attempts = [
        attempt
        for e in result.events
        for attempt in e.details.get("provider_attempts", [])
        if attempt["kind"] == "http_attempt"
    ]
    assert len(attempts) == len(provider.calls)
    assert not any(name in {"supervisor_step", "supervise"} for name, _ in provider.calls)
    assert all(t.requested_by == "orchestrator" for t in result.task_graph)
    assert not any(
        name in {"read_evidence", "synthesize_claims", "check_claims", "write_report"}
        for name, _ in provider.calls
    )
    assert any(
        u.agent == "writer" and u.calls >= 2 and u.execution == "api" for u in result.agent_usage
    )
    assert "EXCLUDED_SOURCE_ONLY" not in str(result.model_dump())


def test_later_search_preserves_ranked_earlier_sources_without_duplicates():
    old = [f"old_{index}" for index in range(12)]
    new = [f"new_{index}" for index in range(12)]
    merged = merge_source_candidates(old, new)
    assert len(merged) == len(set(merged)) == 12
    assert set(old[:6]).issubset(merged) and set(new[:6]).issubset(merged)
    assert merge_source_candidates([], old) == old
    assert merge_source_candidates(old, old) == old


async def test_reader_keeps_first_round_evidence_after_full_second_search(settings, monkeypatch):
    from research_assist.infrastructure.search import HybridSearch, chunk_view

    settings.chunk_chars, settings.chunk_overlap = 300, 40
    requests = []

    async def search(self, queries, document_ids, top_k, mode):
        chunks = self.repository.chunks(document_ids)
        assert len(chunks) >= 24
        # Two query lanes per round. A later round explores entirely new IDs.
        start = 0 if len(requests) < 2 else 12
        requests.append(queries)
        return [chunk_view(c, score=1.0) for c in chunks[start : start + 12]], []

    monkeypatch.setattr(HybridSearch, "search", search)
    provider = AutonomousProvider(search_to_limit=True)
    reader_inputs = []

    def handle(request):
        body = json.loads(request.content)
        if body["response_format"]["json_schema"]["name"] == "reader_step":
            payload = json.loads(body["messages"][1]["content"])
            text = " ".join(u["text"] for u in payload["context"]["source_units"])
            reader_inputs.append(text)
            assert "SCALE_ANCHOR" in text
        response = provider(request)
        if body["response_format"]["json_schema"]["name"] == "fact_checker_step":
            data = response.json()
            output = json.loads(data["choices"][0]["message"]["content"])
            if output.get("result") and len(provider.summary_jobs) == 1:
                output["result"]["coverage"] = [
                    {
                        "requirement_id": "rq_1",
                        "status": "missing",
                        "claim_ids": [],
                        "explanation": "A requested aspect still needs a focused second search.",
                    }
                ]
                data["choices"][0]["message"]["content"] = json.dumps(output)
                response = httpx.Response(200, json=data)
        return response

    result = await run_autonomous(
        settings,
        handle,
        max_calls=24,
        content=(
            "SCALE_ANCHOR RAG evidence explains the dataset comparison. "
            + "RAG evidence describes annotations, conditions and limitations. " * 200
        ).encode(),
    )
    assert len(requests) == 4 and reader_inputs
    assert result.reported_claim_ids


async def test_retrieval_outage_hands_over_observed_sources_and_still_checks_claims(
    settings, monkeypatch
):
    provider = AutonomousProvider()
    failures = []

    async def immediate(_):
        pass

    monkeypatch.setattr("research_assist.infrastructure.llm.asyncio.sleep", immediate)

    def handle(request):
        body = json.loads(request.content)
        task = body["response_format"]["json_schema"]["name"]
        if task == "retrieval_step" and provider.steps["job_1"] >= 1:
            failures.append(task)
            return httpx.Response(503)
        return provider(request)

    result = await run_autonomous(settings, handle, max_calls=24)
    # The local handoff now avoids this redundant API decision altogether.
    assert not failures
    assert result.stop_reason == "requirements_covered"
    assert {s.role for s in result.agent_sessions} == {
        "retrieval",
        "reader",
        "summarizer",
        "fact_checker",
        "writer",
    }
    recovery = [e for e in result.events if e.details.get("recovery") == "observed_sources_only"]
    assert not recovery
    assert result.reported_claim_ids and all(c.verdict == "supported" for c in result.checks)
    assert any(o["kind"] == "handoff" for s in result.agent_sessions for o in s.observations)


@pytest.mark.parametrize("status,after_search", [(503, False), (400, True), (429, True)])
async def test_no_source_or_nontransient_error_cannot_use_source_recovery(
    settings, monkeypatch, status, after_search
):
    provider = AutonomousProvider()

    async def immediate(_):
        pass

    monkeypatch.setattr("research_assist.infrastructure.llm.asyncio.sleep", immediate)

    def handle(request):
        body = json.loads(request.content)
        task = body["response_format"]["json_schema"]["name"]
        if task == ("reader_step" if after_search else "retrieval_step"):
            return httpx.Response(status)
        return provider(request)

    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-only-key")
    settings.orchestration_mode = "autonomous"
    settings.llm_requests_per_minute = 1000
    container = build_container(settings, httpx.MockTransport(handle))
    try:
        document, _ = await container.documents.ingest(
            b"RAG evidence supports claims using passages from the selected source document.",
            "paper.txt",
        )
        record = container.runs.submit(
            ResearchRequest(
                question="Explain RAG evidence",
                document_ids=[document.id],
                scope_mode="selected",
                use_cache=False,
            )
        )
        await container.runs.tasks[record.id]
        completed = container.repository.run(record.id)
        assert completed.status == "failed" and completed.result is None
        assert not any(e.details.get("recovery") for e in completed.events)
    finally:
        await container.close()


async def test_real_world_enum_padding_and_search_limit_preserve_checker_budget(settings):
    provider = AutonomousProvider(padded_arguments=True, search_to_limit=True)
    result = await run_autonomous(settings, provider, max_calls=12)
    assert result.stop_reason == "requirements_covered"
    assert result.reported_claim_ids and result.llm_calls <= 12
    retrieval = next(s for s in result.agent_sessions if s.role == "retrieval")
    assert retrieval.steps == 1
    assert any(o["kind"] == "handoff" for o in retrieval.observations)
    assert not any(o["kind"] == "rejected" for s in result.agent_sessions for o in s.observations)
    assert any(s.role == "fact_checker" and s.status == "completed" for s in result.agent_sessions)


async def test_writer_can_drive_research_through_peer_dependencies_then_resume(settings):
    provider = AutonomousProvider(writer_driven=True)
    result = await run_autonomous(settings, provider, max_calls=30)
    assert result.stop_reason == "requirements_covered"
    parent = next(t for t in result.task_graph if t.role == "writer")
    assert parent.role == "writer" and parent.status == "completed"
    assert len(parent.depends_on) == 2
    assert {t.role for t in result.task_graph if t.parent_id == parent.id} == {
        "summarizer",
        "fact_checker",
    }
    session = next(s for s in result.agent_sessions if s.task_id == parent.id)
    assert session.steps == 3 and session.status == "completed"
    assert sum(o["kind"] == "request_task" for o in session.observations) == 2
    assert result.reported_claim_ids


async def test_budget_after_new_draft_keeps_prior_verified_snapshot(settings):
    provider = AutonomousProvider(budget_after_revision=True)
    result = await run_autonomous(settings, provider, budget=1000)
    assert result.stop_reason == "budget_exhausted"
    assert provider.original_claim in result.markdown
    assert "This proposed revision" not in result.markdown
    assert len(result.reported_claim_ids) == 1
    assert any(s.role == "writer" and s.status == "blocked" for s in result.agent_sessions)


async def test_writer_repairs_invented_claim_id_before_layout_is_accepted(settings):
    provider = AutonomousProvider(invalid_writer=True)
    result = await run_autonomous(settings, provider)
    assert result.stop_reason == "requirements_covered"
    session = next(s for s in result.agent_sessions if s.role == "writer")
    assert any(o["kind"] == "rejected" for o in session.observations)
    assert "cl_invented" not in result.reported_claim_ids
    assert "Unchecked text" not in result.markdown


async def test_early_global_budget_does_not_publish_unchecked_draft(settings):
    result = await run_autonomous(settings, AutonomousProvider(), max_calls=6)
    assert result.stop_reason == "budget_exhausted"
    assert not result.reported_claim_ids and not result.claims
    assert result.llm_calls == 6


async def test_code_scheduler_completes_on_its_last_allowed_dispatch(settings):
    result = await run_autonomous(settings, AutonomousProvider(), max_dispatches=5)
    assert result.stop_reason == "requirements_covered"
    assert len(result.task_graph) == 5


async def test_code_scheduler_limits_dispatches_without_spending_model_calls(settings):
    provider = AutonomousProvider()
    result = await run_autonomous(settings, provider, max_dispatches=1)
    assert result.stop_reason == "coordinator_step_limit"
    assert len(result.task_graph) == 1 and result.task_graph[0].role == "retrieval"
    assert not result.reported_claim_ids
    assert not any(name == "supervisor_step" for name, _ in provider.calls)


async def test_blocked_worker_stops_without_recreating_the_same_task(settings):
    result = await run_autonomous(settings, AutonomousProvider(block_retrieval=True))
    assert result.stop_reason == "task_blocked"
    assert len(result.task_graph) == 1 and result.task_graph[0].status == "blocked"
    assert not result.reported_claim_ids


async def test_changed_draft_invalidates_writer_authority_until_rechecked(settings):
    provider = AutonomousProvider(stale_writer=True)
    result = await run_autonomous(settings, provider, max_calls=30)
    assert result.stop_reason == "task_blocked"
    assert provider.original_claim in result.markdown
    assert "This proposed revision" not in result.markdown
    writer = next(s for s in result.agent_sessions if s.role == "writer")
    assert writer.status == "blocked"
    assert any(
        o["kind"] == "rejected" and "unchecked artifact version" in str(o["data"])
        for o in writer.observations
    )


async def test_section_tool_reads_neighbors_without_crossing_document_scope(settings):
    settings.chunk_chars, settings.chunk_overlap = 350, 60
    container = build_container(settings)
    try:
        doc, _ = await container.documents.ingest(
            ("RAG source evidence has conditions and limitations. " * 50).encode(),
            "long.txt",
        )
        excluded, _ = await container.documents.ingest(
            b"An excluded source must never be read through a section anchor.",
            "other.txt",
        )

        def view(document_id):
            chunk = container.repository.chunks([document_id])[0]
            return RetrievedChunk(**chunk.model_dump(exclude={"vector", "vector_model"}), score=1.0)

        neighbors = await container.search.read_section([view(doc.id)], [doc.id])
        assert 1 < len(neighbors) <= 6
        assert all(c.document_id == doc.id for c in neighbors)
        assert view(doc.id).id in {c.id for c in neighbors}
        with pytest.raises(ProviderError):
            await container.search.read_section([view(excluded.id)], [doc.id])
    finally:
        await container.close()


async def test_section_observation_ids_remain_valid_after_expanded_context_is_preserved(settings):
    settings.chunk_chars, settings.chunk_overlap = 700, 120
    result = await run_autonomous(
        settings,
        AutonomousProvider(),
        content=(
            "RAG retrieved evidence supports generated answers under stated conditions. " * 35
        ).encode(),
    )
    reader = next(s for s in result.agent_sessions if s.role == "reader")
    assert reader.status == "completed" and reader.steps == 2
    assert not any(o["kind"] == "rejected" for o in reader.observations)
    assert result.reported_claim_ids


class GuardProvider(AutonomousProvider):
    """HTTP stub that exercises date translation and structural feedback end to end."""

    def __init__(self, *, date_translation=False, repair=False):
        super().__init__()
        self.date_translation, self.repair = date_translation, repair
        self.feedback = []
        self.number_checks = []

    def __call__(self, request):
        response = super().__call__(request)
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        if name == "question_plan":
            return response
        payload = json.loads(body["messages"][1]["content"])
        context = payload["context"]
        data = response.json()
        output = json.loads(data["choices"][0]["message"]["content"])
        if name == "summarizer_step" and output["action"] == "finish":
            source = context["evidence"][0]
            claim = output["result"]["claims"][0]
            claim["text"] = (
                "Bài báo xuất bản vào ngày 11 tháng 4 năm 2014."
                if self.date_translation
                else source["quote"]
            )
            if not self.date_translation and len(self.summary_jobs) == 1:
                output["result"]["claims"].append(
                    {
                        **claim,
                        "text": "The experiment achieved 99% accuracy.",
                    }
                )
        if name == "fact_checker_step":
            observation = payload["last_observation"]
            self.number_checks.extend(observation.get("number_checks", []))
            if "Structural guards rejected" in observation.get("error", ""):
                self.feedback.append(observation)
                if self.repair:
                    rejected = [c for c in context["checks"] if c["verdict"] != "supported"]
                    output["result"]["checks"] = context["checks"]
                    output["result"]["requests"] = [
                        {
                            "action": "revise",
                            "requirement_ids": ["rq_1"],
                            "claim_ids": [c["claim_id"] for c in rejected],
                            "need": "Remove the unsupported accuracy using only existing evidence.",
                        }
                    ]
        data["choices"][0]["message"]["content"] = json.dumps(output)
        return httpx.Response(200, json=data)


async def test_date_translation_survives_autonomous_tools_checks_and_report(settings):
    provider = GuardProvider(date_translation=True)
    result = await run_autonomous(
        settings,
        provider,
        content=b"RAG evidence. Date of publication:11 April 2014. Magazine issue: MAY 2014.",
    )
    assert result.stop_reason == "requirements_covered"
    assert "11 tháng 4 năm 2014" in result.markdown
    assert result.coverage[0].status == "covered"
    assert provider.number_checks[0]["numeric_guard_passed"]
    assert not provider.number_checks[0]["literal_numbers_present"]
    assert not provider.feedback and result.llm_calls == 9


@pytest.mark.parametrize("repair", [False, True])
async def test_structural_downgrade_gets_one_feedback_opportunity(settings, repair):
    provider = GuardProvider(repair=repair)
    result = await run_autonomous(settings, provider, max_calls=24)
    assert len(provider.feedback) == 1
    assert "99%" not in result.markdown
    assert result.reported_claim_ids
    if repair:
        assert len(provider.summary_jobs) == 2
        assert result.stop_reason == "requirements_covered"
    else:
        assert len(provider.summary_jobs) == 2
        checker = next(s for s in result.agent_sessions if s.role == "fact_checker")
        assert checker.steps == 3 and checker.status == "completed"
        assert result.coverage[0].status == "covered"
        assert any(t.role == "retrieval" and t.requirement_ids for t in result.task_graph)


async def test_last_local_step_keeps_partial_snapshot_without_retry(settings):
    settings.agent_max_steps = 2
    provider = GuardProvider()
    result = await run_autonomous(settings, provider)
    assert not provider.feedback and result.llm_calls <= 12
    assert result.reported_claim_ids and "99%" not in result.markdown
    assert result.coverage[0].status != "covered"


async def test_feedback_global_budget_keeps_verified_snapshot(settings):
    result = await run_autonomous(settings, GuardProvider(), max_calls=9)
    assert result.stop_reason == "budget_exhausted"
    assert result.llm_calls == 9
    assert result.reported_claim_ids and "99%" not in result.markdown
    assert any(c.verdict == "insufficient" for c in result.checks)
