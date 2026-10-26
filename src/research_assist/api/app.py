import asyncio
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from research_assist import __version__
from research_assist.agents.catalog import public_specs
from research_assist.api.upload_progress import UploadProgress
from research_assist.bootstrap import Container, build_container
from research_assist.config import Settings
from research_assist.domain.errors import (
    CapacityError,
    DocumentError,
    NotFoundError,
    ProviderError,
    ResearchError,
)
from research_assist.domain.metadata import EVIDENCE_POLICY
from research_assist.domain.models import ResearchRequest
from research_assist.domain.ports import DiagnosticReranker


class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(max_length=256)


def secure_equal(candidate: str, secret: str) -> bool:
    return hmac.compare_digest(candidate.encode("utf-8"), secret.encode("utf-8"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    web = Path(__file__).resolve().parents[1] / "web"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(settings)
        app.state.container = container
        try:
            if settings.warmup_embedding:
                try:
                    await container.search.warmup()
                except ProviderError:
                    logging.getLogger(__name__).warning(
                        "Embedding warmup unavailable; retry on demand"
                    )
            yield
        finally:
            await container.close()

    app = FastAPI(title="Research Assist · Five-Agent RAG", version=__version__, lifespan=lifespan)
    app.state.upload_progress = UploadProgress()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    def deps(request: Request) -> Container:
        return request.app.state.container  # type: ignore[no-any-return]

    @app.middleware("http")
    async def boundary(request: Request, call_next: RequestResponseEndpoint) -> Response:
        # UI, docs and health contain no private data. All data APIs are protected when enabled.
        path = request.url.path
        secret = settings.access_token.get_secret_value()
        public = {"/api/health", "/api/auth/status", "/api/auth/login"}
        if secret and path.startswith("/api/") and path not in public:
            bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
            cookie = request.cookies.get("ra_session", "")
            if not (secure_equal(bearer, secret) or secure_equal(cookie, secret)):
                return JSONResponse({"detail": "Cần đăng nhập workspace."}, status_code=401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return JSONResponse(
                    {"detail": "Yêu cầu khác origin không được phép."}, status_code=403
                )
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > (settings.max_upload_mb + 1) * 1024 * 1024:
                    return JSONResponse({"detail": "Yêu cầu tải lên quá lớn."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Content-Length không hợp lệ."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if path not in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; font-src 'self'; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'self'"
            )
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ResearchError)
    async def research_error(request: Request, exc: ResearchError) -> JSONResponse:
        status = (
            404
            if isinstance(exc, NotFoundError)
            else 429
            if isinstance(exc, CapacityError)
            else 400
        )
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> dict[str, bool]:
        secret = settings.access_token.get_secret_value()
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        cookie = request.cookies.get("ra_session", "")
        return {
            "required": bool(secret),
            "authenticated": not secret
            or secure_equal(cookie, secret)
            or secure_equal(bearer, secret),
        }

    @app.post("/api/auth/login")
    async def login(body: LoginBody, request: Request) -> Response:
        secret = settings.access_token.get_secret_value()
        if not secret or not secure_equal(body.token, secret):
            return JSONResponse({"detail": "Mã truy cập không đúng."}, status_code=401)
        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            "ra_session",
            secret,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=28800,
        )
        return response

    @app.post("/api/auth/logout")
    async def logout() -> Response:
        response = JSONResponse({"authenticated": False})
        response.delete_cookie("ra_session")
        return response

    @app.get("/api/config")
    async def config(request: Request) -> dict[str, object]:
        container = deps(request)
        return {
            "provider": container.router.name,
            "model": container.router.model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model if container.embedder.fingerprint else None,
            "reranker_model": settings.resolved_reranker_model or None,
            "reranker_provider": settings.resolved_reranker_provider,
            "demo": settings.resolved_reranker_provider != "cohere"
            and all(route["execution"] != "api" for route in container.router.public_routes()),
            "max_upload_mb": settings.max_upload_mb,
            "max_rounds": 2,
            "auth_required": bool(settings.access_token.get_secret_value()),
            "workflow_profile": settings.workflow_profile,
            "orchestration_mode": settings.orchestration_mode,
            "coordinator_max_steps": settings.coordinator_max_steps,
            "agent_max_steps": settings.agent_max_steps,
            "autonomous_max_steps": settings.autonomous_max_steps,
            "autonomous_max_tasks": settings.autonomous_max_tasks,
            "autonomous_parallelism": settings.autonomous_parallelism,
            "autonomous_team_planning": settings.autonomous_team_planning,
            "agent_routes": container.router.public_routes(),
            "cache_enabled": settings.cache_enabled,
            "llm_max_calls": settings.llm_max_calls,
            "chunk_strategy": settings.chunk_strategy,
            "local_device": settings.local_device,
        }

    @app.get("/api/agents")
    async def agent_contracts() -> dict[str, object]:
        return {
            "policy_version": EVIDENCE_POLICY,
            "agents": public_specs(),
            "orchestrator": {
                "max_rounds": 2,
                "mode": settings.orchestration_mode,
                "max_steps": settings.autonomous_max_steps
                if settings.orchestration_mode == "autonomous"
                else settings.coordinator_max_steps,
                "agent_max_steps": settings.agent_max_steps,
                "execution": "local",
                "agent_count": 5,
                "parallelism": settings.autonomous_parallelism,
                "team_planning": settings.autonomous_team_planning,
                "pattern": "concurrent agent tasks with versioned evidence and peer feedback"
                if settings.orchestration_mode == "autonomous"
                else "code scheduling with typed evidence feedback",
            },
        }

    @app.post("/api/cache/clear")
    async def clear_cache(request: Request) -> dict[str, int]:
        return {"deleted": deps(request).repository.clear_cache()}

    @app.get("/api/diagnostics")
    async def diagnostics(request: Request) -> dict[str, object]:
        container = deps(request)
        reranker = container.runs.pipeline.retrieval.reranker
        chunks = container.repository.chunks()
        indexed = sum(
            c.vector_model == container.embedder.fingerprint and bool(c.vector) for c in chunks
        )
        return {
            "documents": len(container.repository.documents()),
            "chunks": len(chunks),
            "indexed_chunks": indexed,
            "embedding_fingerprint": container.embedder.fingerprint,
            "reindex_needed": bool(container.embedder.fingerprint and indexed != len(chunks)),
            "profile": settings.workflow_profile,
            "agent_routes": container.router.public_routes(),
            "cache_enabled": settings.cache_enabled,
            "budget": {
                "calls": settings.llm_max_calls,
                "tokens": settings.llm_max_total_tokens,
                "requests_per_minute": settings.llm_requests_per_minute,
            },
            "active_runs": len(container.runs.tasks),
            "search_cache": container.search.diagnostics(),
            "reranker": {
                "provider": settings.resolved_reranker_provider,
                "model": settings.resolved_reranker_model or None,
                "batch_size": settings.reranker_batch_size,
                "max_length": settings.reranker_max_length,
                "device": settings.reranker_device,
                "failure_policy": settings.reranker_failure_policy,
                "api": (
                    reranker.diagnostics()
                    if settings.resolved_reranker_provider == "cohere"
                    and isinstance(reranker, DiagnosticReranker)
                    else None
                ),
            },
            "pdf_ocr_enabled": settings.pdf_ocr_enabled,
            "policy_version": EVIDENCE_POLICY,
        }

    @app.get("/api/documents")
    async def documents(request: Request) -> dict[str, object]:
        return {"documents": [d.model_dump() for d in deps(request).repository.documents()]}

    @app.post("/api/documents", status_code=201)
    async def upload(
        request: Request,
        file: Annotated[UploadFile, File()],
        upload_id: Annotated[str | None, Header(alias="X-Upload-ID")] = None,
    ) -> dict[str, object]:
        tracker: UploadProgress = request.app.state.upload_progress
        registered = False
        try:
            if upload_id:
                try:
                    upload_id = str(UUID(upload_id))
                except ValueError:
                    raise DocumentError("Mã tải tài liệu không hợp lệ.") from None
                tracker.start(upload_id, file.size or 0)
                registered = True
            data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise DocumentError(f"Tệp vượt {settings.max_upload_mb} MB.")
            document, duplicate = await deps(request).documents.ingest(
                data,
                file.filename or "file",
                on_progress=(
                    (lambda stage, values: tracker.update(upload_id, stage, values))
                    if upload_id
                    else None
                ),
            )
            if upload_id:
                tracker.finish(upload_id)
            return {"document": document.model_dump(), "duplicate": duplicate}
        except Exception as exc:
            if registered and upload_id:
                tracker.finish(
                    upload_id,
                    str(exc) if isinstance(exc, ResearchError) else "Không xử lý được tài liệu.",
                )
            raise
        finally:
            await file.close()

    @app.get("/api/uploads/{upload_id}")
    async def upload_status(upload_id: str, request: Request) -> dict[str, object]:
        tracker: UploadProgress = request.app.state.upload_progress
        status = tracker.get(upload_id)
        if status is None:
            raise NotFoundError("Không tìm thấy tiến độ tải tài liệu.")
        return status

    @app.post("/api/documents/demo")
    async def seed(request: Request) -> dict[str, object]:
        docs = await deps(request).documents.seed()
        return {"documents": [d.model_dump() for d in docs]}

    @app.post("/api/documents/reindex")
    async def reindex(request: Request) -> dict[str, object]:
        container = deps(request)
        if container.runs.busy:
            raise ResearchError("Dừng các phiên nghiên cứu trước khi tạo lại vector.")
        return {"chunks_updated": await container.documents.reindex()}

    @app.post("/api/documents/rebuild")
    async def rebuild(request: Request) -> dict[str, object]:
        container = deps(request)
        if container.runs.busy:
            raise ResearchError("Dừng các phiên nghiên cứu trước khi xây dựng lại chỉ mục.")
        return {"chunks_updated": await container.documents.rebuild()}

    @app.get("/api/documents/{document_id}/chunks")
    async def chunks(document_id: str, request: Request) -> dict[str, object]:
        repository = deps(request).repository
        repository.document(document_id)
        return {
            "chunks": [
                c.model_dump(exclude={"vector", "vector_model"})
                for c in repository.chunks([document_id])
            ]
        }

    @app.get("/api/documents/{document_id}/file")
    async def original(document_id: str, request: Request) -> FileResponse:
        repository = deps(request).repository
        document = repository.document(document_id)
        media = {
            "pdf": "application/pdf",
            "md": "text/plain",
            "txt": "text/plain",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return FileResponse(
            repository.blob(document_id),
            filename=document.filename,
            media_type=media.get(document.kind),
            content_disposition_type="inline",
        )

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def remove(document_id: str, request: Request) -> Response:
        container = deps(request)
        if container.runs.busy or container.documents.lock.locked():
            return JSONResponse(
                {"detail": "Hãy chờ tác vụ tài liệu hoặc dừng phiên nghiên cứu trước khi xóa."},
                status_code=409,
            )
        await container.documents.delete(document_id)
        return Response(status_code=204)

    @app.get("/api/documents/{document_id}/pages/{page_number}/region")
    async def source_region(
        document_id: str,
        page_number: int,
        clip: str,
        request: Request,
    ) -> Response:
        repository = deps(request).repository
        if repository.document(document_id).kind != "pdf":
            raise DocumentError("Chỉ PDF có tọa độ vùng nguồn.")
        from research_assist.infrastructure.source_preview import render_region

        png = await asyncio.to_thread(
            render_region, repository.blob(document_id), page_number, clip
        )
        return Response(png, media_type="image/png")

    @app.post("/api/runs", status_code=202)
    async def create_run(body: ResearchRequest, request: Request) -> dict[str, object]:
        container = deps(request)
        if container.documents.maintenance:
            raise ResearchError("Đang xây dựng lại chỉ mục; hãy chờ hoàn thành.")
        return container.runs.submit(body).model_dump()

    @app.get("/api/runs")
    async def history(request: Request) -> dict[str, object]:
        return {"runs": deps(request).repository.history()}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict[str, object]:
        return deps(request).repository.run(run_id).model_dump()

    @app.get("/api/runs/{run_id}/trace")
    async def trace(run_id: str, request: Request) -> Response:
        record = deps(request).repository.run(run_id)
        return Response(
            json.dumps(
                {
                    "format_version": 1,
                    "run_id": record.id,
                    "status": record.status,
                    "document_ids": record.request.document_ids,
                    "scope_documents": [d.model_dump() for d in record.scope_documents],
                    "corpus_revision": record.corpus_revision,
                    "events": [e.model_dump() for e in record.events],
                },
                ensure_ascii=False,
                indent=2,
            ),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{record.id}-trace.json"'},
        )

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request) -> dict[str, object]:
        return (await deps(request).runs.cancel(run_id)).model_dump()

    @app.get("/api/runs/{run_id}/export")
    async def export(run_id: str, request: Request, format: str = "md") -> Response:
        run = deps(request).repository.run(run_id)
        if not run.result:
            raise ResearchError("Phiên chưa có báo cáo để xuất.")
        if format not in {"md", "json"}:
            raise ResearchError("Định dạng xuất phải là md hoặc json.")
        content = (
            run.result.markdown
            if format == "md"
            else json.dumps(run.result.model_dump(), ensure_ascii=False, indent=2)
        )
        return Response(
            content,
            media_type="text/markdown; charset=utf-8" if format == "md" else "application/json",
            headers={"Content-Disposition": f'attachment; filename="research-{run_id}.{format}"'},
        )

    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web / "index.html")

    return app
