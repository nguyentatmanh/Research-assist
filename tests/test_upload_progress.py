"""Upload telemetry reflects completed extraction and embedding work."""

import asyncio
from uuid import uuid4

import httpx
from pydantic import SecretStr

from research_assist.api.app import create_app
from research_assist.config import Settings


async def test_upload_progress_reports_real_counts_and_finishes(settings: Settings) -> None:
    settings.chunk_strategy = "chars"
    settings.chunk_chars = 300
    settings.chunk_overlap = 0
    settings.embedding_batch_size = 1
    app = create_app(settings)
    entered = [asyncio.Event() for _ in range(3)]
    release = [asyncio.Event() for _ in range(3)]

    class PausedEmbedder:
        fingerprint = "test-vectors"
        calls = 0

        async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
            call = self.calls
            self.calls += 1
            if call < len(entered):
                entered[call].set()
                await release[call].wait()
            await asyncio.sleep(0.05)
            return [[1.0, 0.0] for _ in texts]

    data = b"Research evidence and retrieval methods support a grounded answer. " * 35
    upload_id = str(uuid4())
    async with app.router.lifespan_context(app):
        app.state.container.documents.embedder = PausedEmbedder()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/api/documents",
                    files={"file": ("study.txt", data, "text/plain")},
                    headers={"X-Upload-ID": upload_id},
                )
            )
            try:
                await asyncio.wait_for(entered[0].wait(), 5)
                progress = (await client.get(f"/api/uploads/{upload_id}")).json()
                assert progress["stage"] == "embedding"
                assert progress["size_bytes"] == len(data)
                assert 0 < progress["text_chars"] <= len(data)
                assert progress["chunks"] >= 3
                assert progress["embedded"] == 0
                assert progress["embedding_percent"] == 0
                assert progress["remaining_seconds"] is None

                release[0].set()
                await asyncio.wait_for(entered[1].wait(), 5)
                progress = (await client.get(f"/api/uploads/{upload_id}")).json()
                assert progress["embedded"] == 1
                assert 0 < progress["embedding_percent"] < 100
                assert progress["remaining_seconds"] is None

                release[1].set()
                await asyncio.wait_for(entered[2].wait(), 5)
                progress = (await client.get(f"/api/uploads/{upload_id}")).json()
                assert progress["embedded"] == 2
                assert progress["remaining_seconds"] >= 1
                assert progress["embedding_rate"] > 0
            finally:
                for gate in release:
                    gate.set()
            response = await asyncio.wait_for(task, 5)
            assert response.status_code == 201
            document = response.json()["document"]
            progress = (await client.get(f"/api/uploads/{upload_id}")).json()
            assert progress["stage"] == "completed"
            assert progress["chunks"] == document["chunk_count"]
            assert progress["embedded"] == document["chunk_count"]
            assert progress["embedding_percent"] == 100
            assert progress["error"] is None


async def test_failed_upload_progress_keeps_safe_error(settings: Settings) -> None:
    app = create_app(settings)
    upload_id = str(uuid4())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/documents",
                files={"file": ("empty.txt", b"", "text/plain")},
                headers={"X-Upload-ID": upload_id},
            )
            assert response.status_code == 400
            progress = (await client.get(f"/api/uploads/{upload_id}")).json()
            assert progress["stage"] == "failed"
            assert "rỗng" in progress["error"]
            assert (await client.get(f"/api/uploads/{uuid4()}")).status_code == 404
            invalid = await client.post(
                "/api/documents",
                files={"file": ("study.txt", b"valid text", "text/plain")},
                headers={"X-Upload-ID": "invalid"},
            )
            assert invalid.status_code == 400


async def test_upload_progress_requires_workspace_auth(settings: Settings) -> None:
    settings.access_token = SecretStr("private-workspace")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (await client.get(f"/api/uploads/{uuid4()}")).status_code == 401
