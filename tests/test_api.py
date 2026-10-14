import time

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from research_assist.api.app import create_app
from research_assist.config import Settings
from research_assist.domain.models import ResearchRequest


def complete(client: TestClient, run_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    raise AssertionError("Run did not complete")


def test_full_demo_pipeline_and_exports(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert "Content-Security-Policy" in client.get("/").headers
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.post("/api/documents/demo").status_code == 200
        response = client.post(
            "/api/runs", json={"question": "RAG và Self-RAG khác nhau như thế nào?"}
        )
        assert response.status_code == 202
        record = complete(client, response.json()["id"])
        assert record["status"] == "completed", record
        result = record["result"]
        assert result["llm_calls"] == 0 and result["provider"] == "demo"
        assert {e["agent"] for e in result["events"]} == {
            "retrieval",
            "reader",
            "summarizer",
            "fact_checker",
            "writer",
        }
        assert result["evidence"] and result["checks"]
        assert all(c["verdict"] == "supported" for c in result["checks"])
        exported = client.get(f"/api/runs/{record['id']}/export?format=md")
        assert exported.status_code == 200 and "Nguồn trích dẫn" in exported.text
        exported_json = client.get(f"/api/runs/{record['id']}/export?format=json").json()
        assert exported_json["run_id"] == record["id"]
        assert len(client.get("/api/runs").json()["runs"]) == 1


def test_unanswerable_question_abstains(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.post(
            "/api/documents",
            files={
                "file": (
                    "paper.txt",
                    b"RAG uses retrieved evidence to support generated answers.",
                    "text/plain",
                )
            },
        )
        record = client.post(
            "/api/runs",
            json={"question": "What temperature makes sourdough bread ferment fastest?"},
        ).json()
        finished = complete(client, record["id"])
        assert finished["status"] == "completed"
        assert not finished["result"]["evidence"]
        assert not finished["result"]["claims"]
        assert "Chưa đủ" in finished["result"]["markdown"]


def test_upload_duplicate_delete_and_sql_injection_scope(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        content = b"RAG uses retrieved evidence to support generated answers."
        response = client.post(
            "/api/documents", files={"file": ("../../study.txt", content, "text/plain")}
        )
        assert response.status_code == 201
        document = response.json()["document"]
        assert document["filename"] == "study.txt"
        duplicate = client.post("/api/documents", files={"file": ("second.txt", content)}).json()
        assert duplicate["duplicate"] and duplicate["document"]["id"] == document["id"]
        assert client.get(f"/api/documents/{document['id']}/file").content == content
        assert "vector" not in str(client.get(f"/api/documents/{document['id']}/chunks").json())
        assert (
            client.post(
                "/api/runs",
                json={"question": "Explain RAG evidence", "document_ids": ["' OR 1=1 --"]},
            ).status_code
            == 404
        )
        assert client.delete(f"/api/documents/{document['id']}").status_code == 204
        assert client.get(f"/api/documents/{document['id']}/file").status_code == 404


def test_private_workspace_requires_cookie_and_never_exposes_model_key(settings: Settings) -> None:
    settings.access_token = SecretStr("workspace-secret")
    settings.openai_api_key = SecretStr("model-secret")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/documents").status_code == 401
        assert client.post("/api/auth/login", json={"token": "wrong"}).status_code == 401
        assert client.post("/api/auth/login", json={"token": "Mật khẩu sai"}).status_code == 401
        assert client.post("/api/auth/login", json={"token": "workspace-secret"}).status_code == 200
        config = client.get("/api/config")
        assert config.status_code == 200 and "secret" not in config.text
        assert (
            "HttpOnly"
            in client.post("/api/auth/login", json={"token": "workspace-secret"}).headers[
                "set-cookie"
            ]
        )
        client.post("/api/auth/logout")
        assert client.get("/api/config").status_code == 401
        assert (
            client.get(
                "/api/config", headers={"Authorization": "Bearer workspace-secret"}
            ).status_code
            == 200
        )


def test_cross_origin_mutation_rejected(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/documents/demo", headers={"Origin": "https://evil.example"})
        assert response.status_code == 403
        assert not client.get("/api/documents").json()["documents"]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "short"},
        {"question": "x" * 2001},
        {"question": "Explain RAG evidence", "top_k": 100},
        {"question": "Explain RAG evidence", "max_rounds": 10},
        {"question": "Explain RAG evidence", "temperature": 2},
    ],
)
def test_research_input_contract_rejects_invalid_input(
    settings: Settings, body: dict[str, object]
) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/runs", json=body).status_code == 422


def test_secret_configuration_fails_fast(tmp_path: object) -> None:
    with pytest.raises(ValidationError, match="RA_GEMINI_API_KEY"):
        Settings(llm_provider="gemini", gemini_api_key="", _env_file=None)
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_chars=300, chunk_overlap=300, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(openai_base_url="https://user:password@example.com/v1", _env_file=None)


def test_document_ids_are_deduplicated() -> None:
    assert ResearchRequest(
        question="Explain RAG evidence", document_ids=["a", "a"]
    ).document_ids == ["a"]
