"""Expose only the authenticated endpoints needed by Research Assist."""

import json
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

MAX_REQUEST_BYTES = 1024 * 1024
CHAT_FIELDS = {"model", "messages", "temperature", "max_tokens", "response_format"}


def create_gateway(upstream_port: int, key: str) -> FastAPI:
    if not key:
        raise ValueError("A worker token is required.")
    authorization = ("Bearer " + key).encode()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{upstream_port}",
            timeout=httpx.Timeout(85, connect=5),
            trust_env=False,
        ) as client:
            app.state.serving_client = client
            yield

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)

    def require_key(request: Request) -> None:
        candidate = request.headers.get("Authorization", "").encode()
        if not secrets.compare_digest(candidate, authorization):
            raise HTTPException(401, "Worker token is missing or invalid.")

    async def forward(path: str, method: str, body: dict | None = None) -> Response:
        try:
            reply = await app.state.serving_client.request(method, path, json=body)
        except httpx.TimeoutException:
            raise HTTPException(504, "Model response timed out.") from None
        except httpx.RequestError:
            raise HTTPException(503, "The model server is not available.") from None
        return Response(
            reply.content,
            status_code=reply.status_code,
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        require_key(request)
        return await forward("/v1/models", "GET")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        require_key(request)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Request is too large.")
            raw.extend(chunk)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(400, "Invalid JSON.") from None
        if not isinstance(body, dict) or body.keys() - CHAT_FIELDS:
            raise HTTPException(400, "Only Research Assist chat fields are accepted.")
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not 1 <= len(messages) <= 32
            or any(
                not isinstance(message, dict)
                or message.keys() != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str)
                for message in messages
            )
        ):
            raise HTTPException(400, "Only plain-text chat messages are accepted.")
        return await forward("/v1/chat/completions", "POST", body)

    return app


if __name__ == "__main__":
    import sys

    import uvicorn

    upstream, port = map(int, sys.argv[1:3])
    gateway = create_gateway(upstream, os.environ.pop("RESEARCH_GPU_API_KEY"))
    uvicorn.run(gateway, host="127.0.0.1", port=port, access_log=False)
