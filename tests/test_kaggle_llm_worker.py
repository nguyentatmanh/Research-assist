"""Exercise the worker API boundary and generated notebook without CUDA or a tunnel."""

import ast
import json
from pathlib import Path

import httpx

from notebooks.kaggle_llm_gateway import MAX_REQUEST_BYTES, create_gateway
from scripts.build_kaggle_llm_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def test_generated_worker_notebook_defaults_are_inert():
    book = json.loads((ROOT / "notebooks/06_kaggle_llm_worker.ipynb").read_text(encoding="utf-8"))
    assert book == build_notebook()
    namespace = {}
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        exec(compile(source, cell["id"], "exec"), namespace)
    assert namespace["START_LLM_WORKER"] is False
    assert namespace["OPEN_PUBLIC_TUNNEL"] is False
    assert namespace["STOP_LLM_WORKER"] is False
    ast.parse(namespace["GATEWAY_SOURCE"])


async def test_worker_only_exposes_authenticated_models_and_structured_chat():
    model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    headers = {"Authorization": "Bearer " + "s" * 64}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Read this short claim."}],
        "temperature": 0.1,
        "max_tokens": 128,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verify",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"supported": {"type": "boolean"}},
                    "required": ["supported"],
                    "additionalProperties": False,
                },
            },
        },
    }
    requests = []

    def upstream(request):
        requests.append(request)
        assert request.headers.get("Authorization") is None
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"supported": true}'}}]},
        )

    app = create_gateway(8000, "s" * 64)
    async with (
        httpx.AsyncClient(
            base_url="http://gateway", transport=httpx.ASGITransport(app=app)
        ) as client,
        httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(upstream)
        ) as mock,
    ):
        app.state.serving_client = mock
        catalog = "/v1/models"
        chat = "/v1/chat/completions"
        assert (await client.get(catalog)).status_code == 401
        assert (await client.post(chat, json=body)).status_code == 401
        assert (await client.get("/metrics", headers=headers)).status_code == 404
        unsupported = await client.post(chat, headers=headers, json={**body, "stream": True})
        assert unsupported.status_code == 400
        assert (
            await client.post(
                chat,
                headers=headers,
                json={**body, "messages": [{"role": "user", "content": []}]},
            )
        ).status_code == 400
        oversized = await client.post(chat, headers=headers, content=b"x" * (MAX_REQUEST_BYTES + 1))
        assert oversized.status_code == 413
        assert requests == []
        assert (await client.get(catalog, headers=headers)).json() == {"data": [{"id": model}]}
        result = await client.post(chat, headers=headers, json=body)
        assert result.status_code == 200
        assert json.loads(result.json()["choices"][0]["message"]["content"]) == {"supported": True}
        assert result.headers["Cache-Control"] == "no-store"
