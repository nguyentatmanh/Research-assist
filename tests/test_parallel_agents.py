"""Overlap, peer delivery, stale writes, cancellation and shared accounting."""

import asyncio
import json

import httpx
from pydantic import SecretStr

from research_assist.application.agent_runtime import AgentRuntime
from research_assist.domain.errors import BudgetExceeded
from research_assist.domain.models import AgentBrief, AgentSession, SearchCompletion
from research_assist.infrastructure.llm import HTTPLanguageModel, usage_scope
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import ModelRouter
from tests.test_autonomous import AutonomousProvider, decision, run_autonomous


def response(output, inputs=10):
    return httpx.Response(
        200,
        json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": inputs, "completion_tokens": 5},
        },
    )


async def test_five_plans_overlap_and_feedback_reaches_recipient(settings):
    provider = AutonomousProvider()
    entered, all_entered = set(), asyncio.Event()
    roles = ["retrieval", "reader", "summarizer", "fact_checker", "writer"]

    async def handle(request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        if name.endswith("_brief"):
            role = name.removesuffix("_brief")
            entered.add(role)
            if len(entered) == 5:
                all_entered.set()
            # This cannot succeed if the five role calls are serialized.
            await asyncio.wait_for(all_entered.wait(), 3)
            return response(
                {
                    "approach": "Check source conditions and answer only supported requirements.",
                    "feedback": [
                        {
                            "recipient": roles[(roles.index(role) + 1) % 5],
                            "message": "Check conditions and preserve the exact source evidence.",
                        }
                    ],
                }
            )
        return provider(request)

    settings.autonomous_team_planning = True
    result = await run_autonomous(settings, handle, max_calls=24)
    assert entered == set(roles) and set(result.agent_plans) == set(roles)
    assert len(result.peer_messages) == 5
    reader = next(p for name, p in provider.calls if name == "reader_step")
    assert reader["context"]["peer_feedback"][0]["sender"] == "retrieval"
    assert all(m["recipient"] == "reader" for m in reader["context"]["peer_feedback"])
    assert result.reported_claim_ids and result.stop_reason == "requirements_covered"
    # Concurrent calls must not be counted against multiple roles.
    assert sum(u.calls for u in result.agent_usage) == result.llm_calls
    assert sum(u.input_tokens for u in result.agent_usage) == result.input_tokens
    assert sum(u.output_tokens for u in result.agent_usage) == result.output_tokens
    attempts = [
        a
        for e in result.events
        for a in e.details.get("provider_attempts", [])
        if a["kind"] == "http_attempt"
    ]
    assert len(attempts) == result.llm_calls


async def test_independent_reader_and_retrieval_followups_overlap(settings):
    provider = AutonomousProvider()
    requested = False
    entered, both_entered = set(), asyncio.Event()

    async def handle(request):
        nonlocal requested
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        payload = json.loads(body["messages"][1]["content"])
        assignment = payload.get("context", {}).get("assignment", {})
        if assignment.get("requested_by") == "fact_checker" and name in {
            "retrieval_step",
            "reader_step",
        }:
            entered.add(name)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 3)
        result = provider(request)
        data = result.json()
        output = json.loads(data["choices"][0]["message"]["content"])
        if name == "fact_checker_step" and output["action"] == "finish" and not requested:
            requested = True
            output["result"]["requests"] = [
                {
                    "action": action,
                    "requirement_ids": ["rq_1"],
                    "need": "Independently inspect missing source conditions.",
                    "chunk_ids": [payload["context"]["evidence"][0]["chunk_id"]]
                    if action == "read_context"
                    else [],
                }
                for action in ("retrieve", "read_context")
            ]
        if name == "summarizer_step" and output["action"] == "finish":
            # Keep the follow-up grounded instead of the fixture's deliberately wrong revision.
            output["result"]["claims"][0]["text"] = payload["context"]["evidence"][0]["quote"]
        data["choices"][0]["message"]["content"] = json.dumps(output)
        return httpx.Response(200, json=data)

    result = await run_autonomous(settings, handle, max_calls=30)
    assert entered == {"reader_step", "retrieval_step"}
    assert result.reported_claim_ids and result.stop_reason == "requirements_covered"


async def test_stale_decision_is_rejected_before_applying_result():
    version = 0
    accepted, errors = [], []

    class ChangingModel:
        async def generate(self, **kwargs):
            nonlocal version
            if version == 0:
                version = 1  # Another worker publishes new evidence while this call is in flight.
                return kwargs["schema"].model_validate(
                    decision("finish", result={"chunk_ids": ["stale"]})
                )
            errors.append(kwargs["payload"]["last_observation"]["error"])
            return kwargs["schema"].model_validate(
                decision("finish", result={"chunk_ids": ["current"]})
            )

    async def accept(result):
        accepted.append(result.chunk_ids)

    async def guard(work, expected):
        return await work()

    async def step(agent, message, work, describe=None):
        return await work

    await AgentRuntime(ChangingModel(), 3).run(
        AgentSession(task_id="job_1", role="retrieval"),
        instruction="Inspect current sources.",
        output=SearchCompletion,
        tools=[],
        context=lambda: {},
        revision=lambda: str(version),
        accept=accept,
        request_task=lambda task: "unused",
        fallback=lambda: None,
        step=step,
        event=lambda *args, **kwargs: None,
        guard=guard,
    )
    assert accepted == [["current"]] and "changed" in errors[0]


async def test_feedback_tools_deliver_in_both_directions_without_becoming_evidence(settings):
    provider = AutonomousProvider()
    sent = set()

    async def handle(request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        result = provider(request)
        if name in {"reader_step", "fact_checker_step"} and name not in sent:
            sent.add(name)
            return response(
                decision(
                    "use_tool",
                    tool="send_feedback",
                    arguments={
                        "recipient": "fact_checker" if name == "reader_step" else "reader",
                        "message": "Check source conditions before accepting the proposed claim.",
                    },
                )
            )
        return result

    result = await run_autonomous(settings, handle)
    assert {(m.sender, m.recipient) for m in result.peer_messages} == {
        ("reader", "fact_checker"),
        ("fact_checker", "reader"),
    }
    checker = next(p for name, p in provider.calls if name == "fact_checker_step")
    assert checker["context"]["peer_feedback"][0]["sender"] == "reader"
    assert all(e.quote != result.peer_messages[0].message for e in result.evidence)
    assert result.reported_claim_ids


async def test_budget_failure_cancels_and_drains_inflight_team_calls(settings):
    provider = AutonomousProvider()
    active, cancelled = 0, 0

    async def handle(request):
        nonlocal active, cancelled
        name = json.loads(request.content)["response_format"]["json_schema"]["name"]
        if not name.endswith("_brief"):
            return provider(request)
        active += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    settings.autonomous_team_planning = True
    result = await asyncio.wait_for(run_autonomous(settings, handle, max_calls=3), 5)
    assert active == 0 and cancelled == result.llm_calls == 3
    assert result.stop_reason == "budget_exhausted" and not result.reported_claim_ids


async def test_concurrent_budget_is_rechecked_after_shared_rate_gate(settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-test")
    settings.llm_max_calls = 3
    entered, opened, sent = 0, asyncio.Event(), []

    class Gate:
        async def acquire(self):
            nonlocal entered
            entered += 1
            if entered == 5:
                opened.set()
            await opened.wait()

    def handle(request):
        sent.append(request)
        return response({"approach": "Inspect sources before drawing conclusions.", "feedback": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = HTTPLanguageModel(settings, client, Gate())
        with usage_scope() as usage:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        model.generate(
                            task="retrieval_brief",
                            instruction="Plan only.",
                            payload={},
                            schema=AgentBrief,
                            fallback=lambda: None,
                        )
                        for _ in range(5)
                    ],
                    return_exceptions=True,
                ),
                3,
            )
        assert len(sent) == usage.calls == 3
        assert sum(isinstance(r, BudgetExceeded) for r in results) == 2


async def test_concurrent_routes_attribute_tokens_and_cost_to_the_correct_role(settings, tmp_path):
    settings.llm_provider = "openai"
    settings.openai_api_key = SecretStr("offline-test")
    settings.cache_enabled = False
    settings.orchestration_mode = "autonomous"
    arrived, opened = 0, asyncio.Event()
    tokens = {"reader_brief": 11, "writer_brief": 29}

    async def handle(request):
        nonlocal arrived
        name = json.loads(request.content)["response_format"]["json_schema"]["name"]
        arrived += 1
        if arrived == 2:
            opened.set()
        await opened.wait()
        return response(
            {"approach": "Inspect sources before drawing conclusions.", "feedback": []},
            tokens[name],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(settings, client, Repository(tmp_path))
        with usage_scope(orchestration_mode="autonomous") as usage:
            await asyncio.wait_for(
                asyncio.gather(
                    *[
                        router.generate(
                            task=name,
                            instruction="Plan only.",
                            payload={},
                            schema=AgentBrief,
                            fallback=lambda: None,
                        )
                        for name in tokens
                    ]
                ),
                3,
            )
        assert usage.calls == 2 and usage.input_tokens == 40 and usage.output_tokens == 10
        assert usage.agents["reader"].input_tokens == 11
        assert usage.agents["writer"].input_tokens == 29
        assert all(u.calls == 1 for u in usage.agents.values())
