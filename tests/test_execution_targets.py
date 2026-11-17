"""Per-run selection must never mutate routes, leak keys, or cross concurrent runs."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from research_assist.api.app import create_app
from research_assist.application.target_probe import probe_model
from research_assist.config import ExecutionTarget, Settings
from research_assist.domain.models import ResearchRequest, RetrievalPlan
from research_assist.infrastructure.llm import usage_scope
from research_assist.infrastructure.repository import Repository
from research_assist.infrastructure.routing import ModelRouter


def configuration(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        llm_provider="demo",
        cache_enabled=False,
        execution_targets={
            "cpu": ExecutionTarget(
                provider="ollama", model="local-test", base_url="http://127.0.0.1:11434/v1"
            ),
            "gpu": ExecutionTarget(
                provider="openai",
                model="gpu-test",
                base_url="http://127.0.0.1:8001/v1",
                api_key=SecretStr("private-worker-token"),
            ),
        },
    )


async def test_concurrent_routes_and_credentials_are_isolated(tmp_path: Path) -> None:
    seen = []
    active = 0
    overlap = False

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, overlap
        active += 1
        overlap = overlap or active > 1
        body = json.loads(request.content)
        seen.append((body["model"], request.url.port, request.headers.get("authorization")))
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "queries": [body["model"]],
                                    "facets": [],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    settings = configuration(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(settings, client, Repository(tmp_path))

        async def run(target):
            with usage_scope("academic", False, "autonomous", {"retrieval": target}) as usage:
                result = await router.generate(
                    task="retrieval_plan",
                    instruction="test",
                    payload={},
                    schema=RetrievalPlan,
                    fallback=lambda: RetrievalPlan(queries=["demo-query"]),
                )
                await asyncio.sleep(0)
                return result.queries, usage.agents["retrieval"].model

        results = await asyncio.gather(run("cpu"), run("gpu"), run("demo"))
        assert results == [
            (["local-test"], "local-test"),
            (["gpu-test"], "gpu-test"),
            (["demo-query"], "extractive-demo"),
        ]
        assert overlap
        assert ("gpu-test", 8001, "Bearer private-worker-token") in seen
        assert all(
            token != "Bearer private-worker-token" for _, port, token in seen if port != 8001
        )
        assert router.models["retrieval"].name == "demo"
        assert router.selected_target("retrieval") == "default"
        assert "private-worker-token" not in json.dumps(router.public_targets())
        assert "127.0.0.1" not in json.dumps(router.public_targets())


def test_unconfigured_selection_rejected_and_config_redacted(tmp_path: Path) -> None:
    with TestClient(create_app(configuration(tmp_path))) as client:
        config = client.get("/api/config")
        assert "private-worker-token" not in config.text
        assert "api_key" not in config.text
        targets = {t["id"]: t for t in config.json()["execution_targets"]}
        assert targets["gpu"]["configured"] is True
        assert targets["api"]["configured"] is False
        client.post("/api/documents/demo")
        response = client.post(
            "/api/runs",
            json={
                "question": "How does RAG use retrieved evidence?",
                "agent_targets": {"writer": "api"},
            },
        )
        assert response.status_code == 400
        assert client.get("/api/runs").json()["runs"] == []
        assert client.post("/api/config/targets/unknown/test").status_code == 404
        assert client.post("/api/config/targets/demo/test").json()["reachable"] is True


def test_invalid_roles_targets_and_endpoints_rejected() -> None:
    for selection in ({"router": "demo"}, {"writer": "https://evil.invalid"}):
        with pytest.raises(ValidationError):
            ResearchRequest(question="A valid research question", agent_targets=selection)
    for url in (
        "http://remote.example/v1",
        "https://user:secret@remote.example/v1",
        "https://remote.example/v1?key=secret",
    ):
        with pytest.raises(ValidationError):
            ExecutionTarget(provider="ollama", model="test", base_url=url)
    with pytest.raises(ValidationError):
        ExecutionTarget(provider="openai", model="test", base_url="https://example.org/v1")


@pytest.mark.parametrize(
    "status,payload,reachable,found",
    [
        (200, {"data": [{"id": "gpu-test"}]}, True, True),
        (200, {"data": [{"id": "different-model"}]}, True, False),
        (401, {"error": "private-worker-token"}, False, None),
        (302, {}, False, None),
        (200, ["bad-schema"], False, None),
    ],
)
async def test_probe_is_honest_and_redacts_failures(tmp_path, status, payload, reachable, found):
    settings = configuration(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload))
    ) as client:
        router = ModelRouter(settings, client, Repository(tmp_path))
        result = await probe_model(router.target_routes["gpu"], client)
    assert result["reachable"] is reachable
    assert result["model_found"] is found
    assert "private-worker-token" not in json.dumps(result)


def test_target_editor_persists_private_settings_and_never_echoes_tokens(tmp_path):
    from research_assist.application.execution_settings import target_settings_path

    settings = Settings(_env_file=None, data_dir=tmp_path)
    payload = {
        "provider": "openai",
        "model": "remote-model",
        "base_url": "https://worker.example/v1",
        "api_key": "saved-private-token",
    }
    with TestClient(create_app(settings)) as client:
        saved = client.put("/api/config/targets/gpu", json=payload)
        assert saved.status_code == 200
        assert "saved-private-token" not in saved.text
        assert saved.json()["targets"][0]["key_configured"] is True
        assert target_settings_path(settings).is_file()
        current = client.app.state.container.router
        assert client.app.state.container.runs.pipeline.writer.llm is current
        assert current.target_models["gpu"].model == "remote-model"
        # Preserve the key only at the same provider+endpoint.
        updated = client.put(
            "/api/config/targets/gpu", json={**payload, "model": "another-model", "api_key": ""}
        )
        assert updated.status_code == 200
        assert (
            client.app.state.container.settings.execution_targets["gpu"].api_key.get_secret_value()
            == "saved-private-token"
        )
        rejected = client.put(
            "/api/config/targets/gpu",
            json={**payload, "base_url": "https://elsewhere.example/v1", "api_key": ""},
        )
        assert rejected.status_code == 400
        assert (
            client.app.state.container.settings.execution_targets["gpu"].base_url
            == payload["base_url"]
        )
        bad = client.put("/api/config/targets/gpu", json={**payload, "model": None})
        assert bad.status_code == 422
        assert "saved-private-token" not in bad.text
    # New process-equivalent application loads private settings; browser never resends its key.
    with TestClient(create_app(settings)) as client:
        target = client.get("/api/config/targets").json()["targets"][0]
        assert target["model"] == "another-model"
        assert target["key_configured"] is True
        assert client.delete("/api/config/targets/gpu").status_code == 200
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config/targets").json()["targets"] == []


def test_managed_gpu_ignores_stale_saved_token_and_blocks_edits(tmp_path):
    from research_assist.application.execution_settings import target_settings_path

    target_settings_path(Settings(_env_file=None, data_dir=tmp_path)).write_text(
        json.dumps(
            {
                "gpu": {
                    "provider": "openai",
                    "model": "stale-model",
                    "base_url": "http://localhost:9999/v1",
                    "api_key": "stale-token",
                }
            }
        )
    )
    settings = configuration(tmp_path).model_copy(update={"managed_execution_targets": {"gpu"}})
    with TestClient(create_app(settings)) as client:
        assert client.app.state.container.router.target_models["gpu"].model == "gpu-test"
        assert (
            client.app.state.container.settings.execution_targets["gpu"].api_key.get_secret_value()
            == "private-worker-token"
        )
        assert client.delete("/api/config/targets/gpu").status_code == 400
        assert (
            client.put(
                "/api/config/targets/gpu",
                json={
                    "provider": "ollama",
                    "model": "changed",
                    "base_url": "http://localhost:11434/v1",
                },
            ).status_code
            == 400
        )


def test_target_edit_waits_for_inflight_run(tmp_path):
    import threading

    release = threading.Event()
    settings = configuration(tmp_path)
    with TestClient(create_app(settings)) as client:
        client.post("/api/documents/demo")
        pipeline = client.app.state.container.runs.pipeline
        original = pipeline.run

        async def delayed(*args, **kwargs):
            await asyncio.to_thread(release.wait, 5)
            return await original(*args, **kwargs)

        pipeline.run = delayed
        run = client.post("/api/runs", json={"question": "How does RAG use retrieved evidence?"})
        assert run.status_code == 202
        try:
            blocked = client.put(
                "/api/config/targets/cpu",
                json={
                    "provider": "ollama",
                    "model": "different",
                    "base_url": "http://localhost:11434/v1",
                },
            )
            assert blocked.status_code == 429
            assert client.app.state.container.router.target_models["cpu"].model == "local-test"
        finally:
            release.set()


def test_target_settings_require_auth_and_same_origin(tmp_path):
    settings = configuration(tmp_path).model_copy(
        update={"access_token": SecretStr("a-long-workspace-secret-value")}
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config/targets").status_code == 401
        client.post("/api/auth/login", json={"token": "a-long-workspace-secret-value"})
        payload = {"provider": "ollama", "model": "qwen", "base_url": "http://localhost:11434/v1"}
        assert (
            client.put(
                "/api/config/targets/cpu", json=payload, headers={"origin": "https://other.example"}
            ).status_code
            == 403
        )


def test_run_identity_describes_selected_models():
    from research_assist.application.pipeline import runtime_identity
    from research_assist.domain.models import AgentUsage
    from research_assist.infrastructure.llm import DemoLanguageModel, Usage

    usage = Usage(
        agent_targets={"writer": "gpu"},
        agents={
            "writer": AgentUsage(
                agent="writer", provider="openai", model="gpu-model", execution="api"
            ),
            "reader": AgentUsage(
                agent="reader", provider="demo", model="extractive-demo", execution="demo"
            ),
        },
    )
    provider, model = runtime_identity(DemoLanguageModel(), usage)
    assert provider == "mixed" and "gpu-model" in model


async def test_selected_model_controls_n_way_budget_and_demo_label(tmp_path):
    from research_assist.application.parallel_research import ParallelResearchCoordinator
    from research_assist.bootstrap import build_container

    container = build_container(configuration(tmp_path))
    pipeline = container.runs.pipeline
    team = ParallelResearchCoordinator(
        pipeline.retrieval, pipeline.reader, pipeline.summarizer, pipeline.checker, pipeline.writer
    )
    try:
        with usage_scope(agent_targets={"reader": "gpu", "fact_checker": "cpu"}):
            assert team.model_call_cost("reader") == 1
            assert team.model_call_cost("fact_checker") == 1
            assert team.model_call_cost("summarizer") == 0
            assert not container.router.uses_demo_only()
        with usage_scope(agent_targets={"reader": "demo"}):
            assert team.model_call_cost("reader") == 0
            assert container.router.uses_demo_only()
    finally:
        await container.close()
