import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from research_assist.config import AgentRoute, Settings
from research_assist.domain.errors import BudgetExceeded, ProviderError
from research_assist.domain.models import RetrievalPlan
from research_assist.infrastructure.llm import backoff_delay, usage_scope
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import ModelRouter

PRIMARY = "gemini-3.8-flash"
SECONDARY = "gemini-3.5-flash-lite"


@pytest.fixture
def failover_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    async def immediate(_: float) -> None:
        pass

    monkeypatch.setattr("research_assist.infrastructure.llm.asyncio.sleep", immediate)
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        llm_provider="gemini",
        llm_model=PRIMARY,
        gemini_api_key=SecretStr("unit-key"),
        orchestration_mode="autonomous",
        llm_requests_per_minute=1000,
        agent_routes={
            role: AgentRoute(fallback_model=SECONDARY) for role in ("reader", "summarizer")
        },
    )


def success() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": '{"queries":["Evidence"],"facets":[]}'}]},
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        },
    )


async def invoke(router: ModelRouter, task: str = "reader_step") -> RetrievalPlan:
    return await router.generate(
        task=task,
        instruction="Return queries",
        payload={"question": "Evidence"},
        schema=RetrievalPlan,
        fallback=lambda: RetrievalPlan(queries=["DEMO MUST NOT RUN"]),
    )


async def test_failover_audit_circuit_scope_and_cache(failover_settings: Settings) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert "temperature" not in json.loads(request.content)["generationConfig"]
        if PRIMARY in request.url.path:
            return httpx.Response(503, text="PRIVATE unit-key")
        return success()

    repository = Repository(failover_settings.data_dir)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(failover_settings, client, repository)
        with usage_scope() as usage:
            assert (await invoke(router)).queries == ["Evidence"]
            await invoke(router, "summarizer_step")
            assert usage.calls == 5
            switches = [e for e in usage.provider_events if e["kind"] == "model_fallback"]
            assert [e["primary_skipped"] for e in switches] == [False, True]
            assert usage.agents["reader"].models_used == [PRIMARY, SECONDARY]
            assert usage.agents["summarizer"].models_used == [SECONDARY]
            assert usage.agents["summarizer"].model == SECONDARY
            assert "unit-key" not in json.dumps(usage.provider_events)
            assert router.models["reader"].model == PRIMARY
        # A fresh run can try the primary again; fallback content never enters primary cache.
        with usage_scope() as fresh:
            await invoke(router)
            assert fresh.calls == 4 and fresh.cache_hits == 0
        assert len(seen) == 9
    with repository.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM generation_cache").fetchone()[0] == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429])
async def test_client_or_quota_errors_never_switch_model(
    failover_settings: Settings, status: int
) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(status, text="PRIVATE unit-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(failover_settings, client, Repository(failover_settings.data_dir))
        with usage_scope() as usage, pytest.raises(ProviderError, match=f"HTTP {status}"):
            await invoke(router)
        assert all(PRIMARY in path for path in seen)
        assert not usage.unavailable_models
        assert not any(e["kind"] == "model_fallback" for e in usage.provider_events)


async def test_failover_cannot_bypass_call_budget(failover_settings: Settings) -> None:
    failover_settings.llm_max_calls = 3
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        router = ModelRouter(failover_settings, client, Repository(failover_settings.data_dir))
        with usage_scope() as usage, pytest.raises(BudgetExceeded):
            await invoke(router)
        assert usage.calls == 3
        assert usage.agents["reader"].models_used == [PRIMARY]


async def test_timeout_then_secondary_failure_is_bounded(failover_settings: Settings) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if PRIMARY in request.url.path:
            raise httpx.ReadTimeout("PRIVATE unit-key", request=request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(failover_settings, client, Repository(failover_settings.data_dir))
        with usage_scope() as usage, pytest.raises(ProviderError, match="HTTP 503"):
            await invoke(router)
        assert usage.calls == len(seen) == 6
        assert sum(SECONDARY in path for path in seen) == 3


async def test_schema_errors_do_not_trigger_failover(failover_settings: Settings) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        body = success().json()
        body["candidates"][0]["content"]["parts"][0]["text"] = "invalid JSON"
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(failover_settings, client, Repository(failover_settings.data_dir))
        with usage_scope() as usage, pytest.raises(ProviderError, match="hợp đồng"):
            await invoke(router)
        assert usage.calls == 2
        assert not any(e["kind"] == "model_fallback" for e in usage.provider_events)


def test_backoff_grows_with_bounded_jitter() -> None:
    for attempt in range(3):
        delay = backoff_delay(attempt)
        assert 2 * 2**attempt <= delay <= 2.5 * 2**attempt
