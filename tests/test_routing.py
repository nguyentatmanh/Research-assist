import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from research_assist.config import AgentRoute, Settings
from research_assist.domain.errors import ProviderError
from research_assist.domain.models import RetrievalPlan
from research_assist.infrastructure.llm import HTTPLanguageModel, retry_delay, usage_scope
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import ModelRouter


async def test_route_cache_invalidation_and_bypass(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        llm_provider="openai",
        openai_api_key=SecretStr("unit-key"),
        workflow_profile="academic",
        agent_routes={
            "retrieval": AgentRoute(
                model="planner-primary", input_usd_per_million=1, output_usd_per_million=2
            )
        },
        _env_file=None,
    )
    models = []

    def handle(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"queries":["Evidence"],"facets":[]}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(settings, client, Repository(tmp_path))

        async def invoke() -> RetrievalPlan:
            return await router.generate(
                task="retrieval_plan",
                instruction="Plan",
                payload={"question": "Find evidence"},
                schema=RetrievalPlan,
                fallback=lambda: RetrievalPlan(queries=["Fallback"]),
            )

        with usage_scope("academic") as first:
            await invoke()
        assert first.calls == 1
        assert first.agents["retrieval"].estimated_cost_usd == pytest.approx(0.00002)
        with usage_scope("academic") as cached:
            await invoke()
        assert cached.calls == 0 and cached.cache_hits == 1
        with usage_scope("academic", cache_enabled=False) as bypassed:
            await invoke()
        assert bypassed.calls == 1
        router.routes["retrieval"].llm_model = "planner-alternate"
        router.models["retrieval"].model = "planner-alternate"
        with usage_scope("academic") as changed:
            await invoke()
        assert changed.calls == 1
        assert models == ["planner-primary", "planner-primary", "planner-alternate"]


async def test_token_guard_and_deepseek_json_mode(settings: Settings) -> None:
    settings.llm_provider = "deepseek"
    settings.deepseek_api_key = SecretStr("unit-key")
    settings.llm_max_total_tokens = 1000
    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"queries":["Evidence"],"facets":[]}'}}],
                "usage": {"prompt_tokens": 990, "completion_tokens": 10},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = HTTPLanguageModel(settings, client)
        with usage_scope() as usage:
            for i in range(2):
                work = model.generate(
                    task="retrieval_plan",
                    instruction="Plan",
                    payload={},
                    schema=RetrievalPlan,
                    fallback=lambda: RetrievalPlan(queries=["Fallback"]),
                )
                if i:
                    with pytest.raises(ProviderError, match="token"):
                        await work
                else:
                    await work
            assert usage.calls == 1
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["thinking"] == {"type": "disabled"}


async def test_gemini_three_sends_thinking_level(settings: Settings) -> None:
    settings.llm_provider = "gemini"
    settings.llm_model = "gemini-3.5-flash-lite"
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["generationConfig"])
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": '{"queries":["Evidence"],"facets":[]}'}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await HTTPLanguageModel(settings, client).generate(
            task="retrieval_plan",
            instruction="Plan",
            payload={},
            schema=RetrievalPlan,
            fallback=lambda: RetrievalPlan(queries=["Fallback"]),
        )
    assert seen[0]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_retry_after_and_cache_expiry(settings: Settings) -> None:
    assert retry_delay(httpx.Response(429, headers={"Retry-After": "2"}), 0) == 2
    assert retry_delay(httpx.Response(429, headers={"Retry-After": "999"}), 0) == 60
    repository = Repository(settings.data_dir)
    repository.cache_put("expired", "{}", -1, 10)
    assert repository.cache_get("expired") is None
    repository.cache_put("valid", "{}", 60, 10)
    assert repository.cache_get("valid") == "{}"
    assert repository.clear_cache() == 1
