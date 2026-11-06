"""Runtime GPU configuration, secret redaction, persistence and maintenance guards."""

import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from research_assist.api.app import create_app

KEY = "gpu-private-test-" + "x" * 32
MODEL = "research-e5-" + "a" * 32
QWEN_MODEL = "research-qwen3-" + "b" * 32


def test_qwen_endpoint_can_be_saved_and_restored(settings, monkeypatch):
    from research_assist.api import app as module

    original = module.build_container

    def provider(request):
        if request.headers.get("authorization") != "Bearer " + KEY:
            return httpx.Response(401)
        if request.url.path == "/v1/info":
            return httpx.Response(200, json={
                "protocol": "research-embedding-v1", "family": "qwen3",
                "model": QWEN_MODEL, "dimension": 2560,
                "max_seq_length": 512, "normalized": True,
                "tokenizer_model": "Qwen/Qwen3-Embedding-4B",
            })
        return httpx.Response(200, json={"model": QWEN_MODEL, "data": [
            {"index": 0, "embedding": [1.0] + [0.0] * 2559}
        ]})

    monkeypatch.setattr(
        module, "build_container",
        lambda config: original(config, transport=httpx.MockTransport(provider)),
    )
    with TestClient(create_app(settings)) as client:
        selected = body(model_family="qwen3", tokenizer_model="Qwen/Qwen3-Embedding-4B")
        saved = client.put("/api/embedding/remote", json=selected)
        assert saved.status_code == 200, saved.text
        assert saved.json()["model_family"] == "qwen3"
        assert saved.json()["model_id"] == QWEN_MODEL
        wrong = client.post("/api/embedding/remote/test", json={
            **selected, "tokenizer_model": "Qwen/Qwen3-Embedding-0.6B"
        })
        assert wrong.status_code == 400
        assert client.delete("/api/embedding/remote").status_code == 200


@pytest.fixture
def gpu_provider(monkeypatch):
    from research_assist.api import app as module

    original = module.build_container
    calls = []

    def provider(request):
        calls.append(request)
        if request.headers.get("authorization") != "Bearer " + KEY:
            return httpx.Response(401, json={"detail": "Invalid key"})
        if request.url.path == "/v1/info":
            return httpx.Response(
                200,
                json={
                    "protocol": "research-embedding-v1",
                    "family": "e5",
                    "model": MODEL,
                    "dimension": 2,
                    "max_seq_length": 512,
                    "normalized": True,
                },
            )
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "data": [
                    {"index": i, "embedding": [0.6, 0.8]} for i, _ in enumerate(payload["input"])
                ],
            },
        )

    monkeypatch.setattr(
        module,
        "build_container",
        lambda settings: original(settings, transport=httpx.MockTransport(provider)),
    )
    return calls


def body(**changes):
    return {"base_url": "https://gpu.test", "api_key": KEY, **changes}


def test_probe_save_hot_apply_and_restart_never_return_key(settings, gpu_provider):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/embedding/remote").json()["active"] is False
        tested = client.post("/api/embedding/remote/test", json=body())
        assert tested.status_code == 200 and tested.json()["model_id"] == MODEL, tested.text
        assert KEY not in tested.text
        assert not (settings.data_dir / "remote-embedding.json").exists()
        container = client.app.state.container
        old_search = container.search
        saved = client.put("/api/embedding/remote", json=body())
        assert saved.status_code == 200, saved.text
        assert saved.json()["active"] and saved.json()["key_configured"]
        assert saved.json()["base_url"] == "https://gpu.test/v1"
        assert KEY not in saved.text
        assert container.search is not old_search
        assert container.runs.pipeline.retrieval.search is container.search
        assert container.documents.embedder is container.embedder
        assert client.get("/api/config").json()["embedding_provider"] == "remote"
        assert KEY not in client.get("/api/config").text
        assert KEY not in client.get("/api/embedding/remote").text
        persisted = json.loads((settings.data_dir / "remote-embedding.json").read_text("utf-8"))
        assert persisted["embedding_api_key"] == KEY
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/embedding/remote").json()
        assert status["active"] and status["saved"] and status["model_id"] == MODEL
        assert KEY not in json.dumps(status)
        assert client.post("/api/embedding/remote/test", json=body(api_key="")).status_code == 200
        restored = client.delete("/api/embedding/remote")
        assert restored.status_code == 200 and not restored.json()["active"]
        assert not (settings.data_dir / "remote-embedding.json").exists()
        assert client.get("/api/config").json()["embedding_provider"] == settings.embedding_provider
    with TestClient(create_app(settings)) as client:
        assert not client.get("/api/embedding/remote").json()["active"]


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key": "wrong-" + "a" * 32},
        {"model_id": "research-e5-other"},
        {"base_url": "http://outside.test"},
        {"base_url": "https://gpu.test/v1/embeddings"},
        {"base_url": "https://user:password@gpu.test"},
        {"batch_size": 33},
    ],
)
def test_invalid_configuration_is_not_persisted_or_applied(settings, gpu_provider, changes):
    with TestClient(create_app(settings)) as client:
        response = client.put("/api/embedding/remote", json=body(**changes))
        assert response.status_code in {400, 422}
        assert KEY not in response.text
        assert not client.get("/api/embedding/remote").json()["active"]
        assert not (settings.data_dir / "remote-embedding.json").exists()


def test_saved_key_is_not_forwarded_to_a_changed_url_without_reentry(settings, gpu_provider):
    with TestClient(create_app(settings)) as client:
        assert client.put("/api/embedding/remote", json=body()).status_code == 200
        before = len(gpu_provider)
        response = client.post(
            "/api/embedding/remote/test", json=body(base_url="https://different.test", api_key="")
        )
        assert response.status_code == 400
        assert len(gpu_provider) == before
        assert client.get("/api/embedding/remote").json()["base_url"] == "https://gpu.test/v1"


def test_validation_and_protected_routes_do_not_echo_submitted_secrets(settings, gpu_provider):
    settings.access_token = SecretStr("workspace-key")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/embedding/remote").status_code == 401
        assert client.put("/api/embedding/remote", json=body()).status_code == 401
        client.post("/api/auth/login", json={"token": "workspace-key"})
        response = client.post("/api/embedding/remote/test", json=body(batch_size="bad"))
        assert response.status_code == 422 and KEY not in response.text


@pytest.mark.parametrize("busy_kind", ["run", "documents"])
def test_save_and_restore_are_blocked_during_work(settings, gpu_provider, busy_kind):
    with TestClient(create_app(settings)) as client:
        container = client.app.state.container
        if busy_kind == "run":
            container.runs.tasks["fake"] = None
        else:
            container.documents.maintenance = True
        try:
            assert client.put("/api/embedding/remote", json=body()).status_code == 429
            assert client.delete("/api/embedding/remote").status_code == 429
            assert not gpu_provider
        finally:
            container.runs.tasks.clear()
            container.documents.maintenance = False


def test_hot_applied_endpoint_is_used_for_upload_and_rebuild(settings, gpu_provider, monkeypatch):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    monkeypatch.setattr(
        "research_assist.infrastructure.token_chunker.tokenizer_for", lambda *args: Tokenizer()
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/documents/demo").status_code == 200
        saved = client.put("/api/embedding/remote", json=body()).json()
        assert saved["reindex_needed"] and saved["document_count"] == 3
        assert client.post("/api/documents/rebuild").status_code == 200
        status = client.get("/api/embedding/remote").json()
        assert not status["reindex_needed"]
        upload = client.post(
            "/api/documents",
            files={
                "file": (
                    "paper.txt",
                    b"Research evidence is supported by original document passages.",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201
        chunks = client.app.state.container.repository.chunks()
        assert all(c.vector_model == "remote:" + MODEL + ":normalized:v1" for c in chunks)
        assert all(c.vector == [0.6, 0.8] for c in chunks)


def test_save_rechecks_idle_after_remote_probe(settings, monkeypatch):
    from research_assist.api import app as module

    original_build = module.build_container
    holder = {}

    def provider(request):
        holder["container"].documents.maintenance = True
        if request.url.path.endswith("/info"):
            return httpx.Response(
                200,
                json={
                    "protocol": "research-embedding-v1",
                    "family": "e5",
                    "model": MODEL,
                    "dimension": 2,
                    "max_seq_length": 512,
                    "normalized": True,
                },
            )
        return httpx.Response(
            200, json={"model": MODEL, "data": [{"index": 0, "embedding": [0.6, 0.8]}]}
        )

    def build(cfg):
        container = original_build(cfg, transport=httpx.MockTransport(provider))
        holder["container"] = container
        return container

    monkeypatch.setattr(module, "build_container", build)
    with TestClient(create_app(settings)) as client:
        try:
            response = client.put("/api/embedding/remote", json=body())
            assert response.status_code == 429
            assert not client.get("/api/embedding/remote").json()["active"]
            assert not (settings.data_dir / "remote-embedding.json").exists()
        finally:
            holder["container"].documents.maintenance = False


@pytest.mark.parametrize("url", ["https://[broken", "https://gpu.test:bad"])
def test_malformed_urls_fail_safely_without_echoing_secret(settings, gpu_provider, url):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/embedding/remote/test", json=body(base_url=url))
        assert response.status_code == 400 and KEY not in response.text
        assert not gpu_provider
