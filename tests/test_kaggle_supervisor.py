import pytest
from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from kaggle import kaggle_supervisor_gateway as gateway
from kaggle import run_kaggle as launcher

TOKEN = "test-kaggle-token_" + "t" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def supervisor_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RA_ACCESS_TOKEN", TOKEN)
    monkeypatch.setattr(gateway, "RUN_DIR", tmp_path)
    monkeypatch.setattr(gateway, "_active_mode", "hybrid")


def test_launcher_leaves_unsupplied_optional_credentials_to_project_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("RA_KAGGLE_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("RA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RA_OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(launcher, "secret", lambda name: TOKEN if name == "RA_ACCESS_TOKEN" else "")
    monkeypatch.setattr(launcher, "check_loopback_port", lambda _: None)
    monkeypatch.setattr(launcher, "install_dependencies", lambda: None)
    monkeypatch.setattr(launcher, "wait_for_gateway", lambda *_: None)
    monkeypatch.setattr(launcher, "stop_process", lambda _: None)

    class Process:
        def wait(self):
            return 0

    def start(*_, env, **__):
        assert env["RA_ACCESS_TOKEN"] == TOKEN
        assert "RA_GEMINI_API_KEY" not in env
        assert "RA_OPENALEX_API_KEY" not in env
        assert "RA_LLM_PROVIDER" not in env
        return Process()

    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    launcher.run("workspace", public_tunnel=False)


async def test_gateway_checks_credentials_before_routing_gpu_or_admin():
    async with AsyncClient(
        transport=ASGITransport(app=gateway.app), base_url="http://localhost"
    ) as c:
        for method, path, body in (
            ("GET", "/api/admin/logs", None),
            ("POST", "/api/admin/switch_mode", {"mode": "workspace"}),
            ("GET", "/api/admin/vram", None),
            ("POST", "/api/admin/vram/clear", None),
            ("POST", "/api/admin/worker/reset", None),
            ("GET", "/v1/info", None),
            ("POST", "/v1/embeddings", {"input": ["document"]}),
        ):
            assert (await c.request(method, path, json=body)).status_code == 401
            wrong = {"Authorization": "Bearer not-the-token", "Cookie": f"ra_session={TOKEN}"}
            assert (await c.request(method, path, json=body, headers=wrong)).status_code == 401


async def test_gateway_tails_only_known_logs_and_redacts_token(tmp_path):
    path = tmp_path / "workspace.log"
    lines = ["too far back"] * 8_000 + [f"legacy access token {TOKEN}", "last log entry"]
    path.write_text("\n".join(lines), encoding="utf-8")
    async with AsyncClient(
        transport=ASGITransport(app=gateway.app), base_url="http://localhost"
    ) as c:
        tail = await c.get("/api/admin/logs?lines=2", headers=HEADERS)
        invalid = await c.get("/api/admin/logs?service=unknown&lines=2", headers=HEADERS)
        unbounded = await c.get("/api/admin/logs?lines=201", headers=HEADERS)
    assert tail.status_code == 200
    assert tail.json()["logs"] == "legacy access token [redacted]\nlast log entry"
    assert tail.headers["cache-control"] == "no-store"
    assert invalid.status_code == unbounded.status_code == 422


async def test_gateway_preserves_workspace_origin_and_routes_authenticated_worker_requests(
    monkeypatch,
):
    requests = []

    async def handle(request: Request) -> Response:
        requests.append(request)
        if request.url.path == "/api/auth/login":
            await request.aread()
            return Response(200, headers={"Set-Cookie": "ra_session=opaque; HttpOnly; Secure"})
        return Response(200, json={"model": "research-e5-fixture"})

    async with AsyncClient(transport=MockTransport(handle)) as upstream:
        monkeypatch.setattr(gateway.app.state, "upstream", upstream, raising=False)
        async with AsyncClient(
            transport=ASGITransport(app=gateway.app),
            base_url="https://sample.trycloudflare.com",
        ) as c:
            login = await c.post(
                "/api/auth/login",
                json={"token": TOKEN},
                headers={"Origin": str(c.base_url).rstrip("/")},
            )
            info = await c.get("/v1/info", headers=HEADERS)
            clear = await c.post("/api/admin/vram/clear", headers=HEADERS)
    assert login.status_code == info.status_code == clear.status_code == 200
    assert "HttpOnly; Secure" in login.headers["set-cookie"]
    assert requests[0].url.port == gateway.PORT_WORKSPACE
    assert requests[0].headers["host"] == "sample.trycloudflare.com"
    assert requests[0].headers["origin"] == "https://sample.trycloudflare.com"
    assert requests[0].headers["x-forwarded-proto"] == "https"
    assert [req.url.port for req in requests[1:]] == [gateway.PORT_WORKER] * 2
    assert [req.headers["authorization"] for req in requests[1:]] == [HEADERS["Authorization"]] * 2
    assert not any("cookie" in req.headers for req in requests[1:])


async def test_switch_mode_preserves_old_service_if_target_never_gets_ready(monkeypatch):
    from fastapi import HTTPException

    running = {"workspace"}
    monkeypatch.setattr(gateway, "_active_mode", "workspace")
    monkeypatch.setattr(gateway, "is_process_running", running.__contains__)
    monkeypatch.setattr(gateway, "spawn_service", running.add)
    monkeypatch.setattr(gateway, "stop_process", running.remove)
    monkeypatch.setattr(gateway.app.state, "upstream", None, raising=False)

    async def unavailable(*_):
        raise HTTPException(503)

    monkeypatch.setattr(gateway, "wait_for_services", unavailable)
    async with AsyncClient(
        transport=ASGITransport(app=gateway.app), base_url="http://localhost"
    ) as c:
        response = await c.post(
            "/api/admin/switch_mode", json={"mode": "gpu_worker"}, headers=HEADERS
        )
    assert response.status_code == 503
    assert running == {"workspace"}
    assert gateway._active_mode == "workspace"
    assert not gateway._switching


async def test_reset_requires_idle_worker_then_checks_replacement_readiness(monkeypatch):
    lifecycle = []
    monkeypatch.setattr(gateway, "is_process_running", lambda _: True)
    monkeypatch.setattr(
        gateway, "stop_process", lambda service: lifecycle.append(("stop", service))
    )
    monkeypatch.setattr(
        gateway, "spawn_service", lambda service: lifecycle.append(("start", service))
    )

    async def verify(_client, services):
        lifecycle.append(("ready", services))

    monkeypatch.setattr(gateway, "wait_for_services", verify)
    idle = False

    def handle(req: Request) -> Response:
        assert req.url.path == "/api/admin/worker/quiesce"
        assert req.headers["authorization"] == HEADERS["Authorization"]
        return Response(200 if idle else 409)

    async with AsyncClient(transport=MockTransport(handle)) as upstream:
        monkeypatch.setattr(gateway.app.state, "upstream", upstream, raising=False)
        async with AsyncClient(
            transport=ASGITransport(app=gateway.app), base_url="http://localhost"
        ) as c:
            busy = await c.post("/api/admin/worker/reset", headers=HEADERS)
            assert busy.status_code == 409 and not lifecycle
            idle = True
            reset = await c.post("/api/admin/worker/reset", headers=HEADERS)
    assert reset.status_code == 200
    assert lifecycle == [
        ("stop", "gpu_worker"),
        ("start", "gpu_worker"),
        ("ready", ("gpu_worker",)),
    ]


@pytest.fixture
def worker():
    pytest.importorskip("torch")
    from kaggle import kaggle_server

    return kaggle_server


async def test_worker_reports_actual_cuda_free_and_clears_only_when_idle(monkeypatch, worker):
    from threading import Lock

    gpu = worker.torch.cuda
    gib = 1024**3
    lock = Lock()
    cache = {"reserved": 6 * gib}
    monkeypatch.setattr(worker, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(worker, "GPU_COUNT", 1)
    monkeypatch.setattr(worker, "_locks", {"cuda:0": lock})
    monkeypatch.setattr(gpu, "is_initialized", lambda: True)
    monkeypatch.setattr(gpu, "mem_get_info", lambda _: (11 * gib - cache["reserved"], 16 * gib))
    monkeypatch.setattr(gpu, "get_device_name", lambda _: "Tesla T4 stub")
    monkeypatch.setattr(gpu, "memory_allocated", lambda _: 4 * gib)
    monkeypatch.setattr(gpu, "memory_reserved", lambda _: cache["reserved"])
    monkeypatch.setattr(gpu, "memory_stats", lambda _: {"num_ooms": 2})
    monkeypatch.setattr(gpu, "empty_cache", lambda: cache.update(reserved=4 * gib))

    async with AsyncClient(
        transport=ASGITransport(app=worker.app), base_url="http://localhost"
    ) as c:
        assert (await c.get("/api/admin/vram")).status_code == 401
        with lock:
            busy = await c.post("/api/admin/vram/clear", headers=HEADERS)
        clear = await c.post("/api/admin/vram/clear", headers=HEADERS)
    assert busy.status_code == 409
    assert clear.status_code == 200
    before = clear.json()["before"]["devices"][0]
    after = clear.json()["after"]["devices"][0]
    assert before["vram_free_gb"] == 5 and after["vram_free_gb"] == 7
    assert before["vram_allocated_gb"] == after["vram_allocated_gb"] == 4
    assert before["vram_reserved_gb"] == 6 and after["vram_reserved_gb"] == 4
    assert after["oom_count"] == 2


async def test_worker_speaks_research_embedding_protocol_and_legacy_input(monkeypatch, worker):
    from types import SimpleNamespace

    inputs_seen = []
    identity = "research-e5-current"
    monkeypatch.setattr(
        worker,
        "_embedding_info",
        {
            "model": identity,
            "protocol": "research-embedding-v1",
            "family": "e5",
            "normalized": True,
        },
    )

    def encode(texts, **_):
        inputs_seen.extend(texts)
        return SimpleNamespace(tolist=lambda: [[1.0, 0.0]] * len(texts))

    monkeypatch.setattr(worker, "load_embedding_model", lambda: SimpleNamespace(encode=encode))
    async with AsyncClient(
        transport=ASGITransport(app=worker.app), base_url="http://localhost"
    ) as c:
        denied = await c.get("/v1/info")
        info = await c.get("/v1/info", headers=HEADERS)
        response = await c.post(
            "/v1/embeddings",
            json={"input": ["needle"], "input_type": "query", "model": identity},
            headers=HEADERS,
        )
        legacy = await c.post("/api/embed", json={"texts": ["document"]}, headers=HEADERS)
        wrong_model = await c.post(
            "/v1/embeddings",
            json={"input": ["needle"], "input_type": "query", "model": "research-e5-old"},
            headers=HEADERS,
        )
        no_llm = await c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers=HEADERS,
        )
    assert denied.status_code == 401 and wrong_model.status_code == 409
    assert info.json()["model"] == identity and info.json()["protocol"] == "research-embedding-v1"
    assert response.json()["data"] == [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}]
    assert response.json()["model"] == identity and legacy.json()["embeddings"] == [[1.0, 0.0]]
    assert inputs_seen == ["query: needle", "passage: document"]
    assert no_llm.status_code == 503
