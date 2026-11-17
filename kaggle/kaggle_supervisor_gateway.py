"""Loopback supervisor for the optional public Kaggle tunnel."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kaggle.processes import stop_owned_process
from kaggle.security import ALLOWED_HOSTS, LOOPBACK, require_bearer, valid_access_token

PROJECT_DIR = Path(__file__).resolve().parents[1]
PORT_WORKSPACE = int(os.environ.get("WORKSPACE_PORT", "8001"))
PORT_WORKER = int(os.environ.get("WORKER_PORT", "8002"))
RUN_DIR = Path(os.environ.get("RA_KAGGLE_RUN_DIR", "/kaggle/temp/research-assist-master"))
LOG_NAMES = {
    "workspace": "workspace.log",
    "gpu_worker": "gpu_worker.log",
    "supervisor": "supervisor.log",
}
MODE_SERVICES = {
    "workspace": ("workspace",),
    "gpu_worker": ("gpu_worker",),
    "hybrid": ("workspace", "gpu_worker"),
}
SERVICE_HEALTH = {
    "workspace": (PORT_WORKSPACE, "/api/health"),
    "gpu_worker": (PORT_WORKER, "/health"),
}
GPU_ROUTES = {
    "api/embed",
    "api/rerank",
    "v1/info",
    "v1/embeddings",
    "v1/models",
    "v1/chat/completions",
}
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
LOG_TAIL_BYTES = 64 * 1024
MAX_GPU_BODY = 2 * 1024 * 1024

_active_mode = "workspace"
_switching = False
_gpu_requests = 0
_processes: dict[str, subprocess.Popen[bytes]] = {}


def spawn_service(service: str) -> None:
    current = _processes.get(service)
    if current is not None and current.poll() is None:
        return
    port, _ = SERVICE_HEALTH[service]
    if service == "workspace":
        command = [sys.executable, "-m", "research_assist.cli", "serve"]
    else:
        command = [sys.executable, "-m", "uvicorn", "kaggle.kaggle_server:app"]
    pythonpath = os.pathsep.join((str(PROJECT_DIR / "src"), str(PROJECT_DIR)))
    env = {**os.environ, "PYTHONPATH": pythonpath}
    env["RA_ALLOWED_HOSTS"] = json.dumps(ALLOWED_HOSTS)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / LOG_NAMES[service]).open("ab") as log:
        _processes[service] = subprocess.Popen(
            [*command, "--host", LOOPBACK, "--port", str(port)],
            cwd=PROJECT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def stop_process(service: str) -> None:
    process = _processes.get(service)
    if process is not None:
        stop_owned_process(process)
    _processes.pop(service, None)


def is_process_running(service: str) -> bool:
    process = _processes.get(service)
    return process is not None and process.poll() is None


async def is_service_healthy(client: httpx.AsyncClient, service: str) -> bool:
    port, path = SERVICE_HEALTH[service]
    try:
        response = await client.get(f"http://{LOOPBACK}:{port}{path}", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _active_mode
    _active_mode = os.environ.get("INITIAL_MODE", "workspace")
    if _active_mode not in MODE_SERVICES:
        raise RuntimeError("INITIAL_MODE phải là workspace, gpu_worker hoặc hybrid.")
    if not valid_access_token(os.environ.get("RA_ACCESS_TOKEN", "")):
        raise RuntimeError(
            "RA_ACCESS_TOKEN phải có 32–256 ký tự gồm chữ, số, gạch ngang, gạch dưới."
        )
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        app.state.upstream = client
        try:
            for service in MODE_SERVICES[_active_mode]:
                spawn_service(service)
            yield
        finally:
            for service in tuple(_processes):
                await asyncio.to_thread(stop_process, service)


app = FastAPI(title="Research Assist Kaggle Supervisor", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
Admin = Annotated[None, Depends(require_bearer)]


@app.get("/health")
async def gateway_health(request: Request) -> JSONResponse:
    services = MODE_SERVICES[_active_mode]
    status = await asyncio.gather(
        *(is_service_healthy(request.app.state.upstream, service) for service in services)
    )
    ready = all(status) and not _switching
    return JSONResponse(
        {
            "status": "online" if ready else "starting_or_degraded",
            "active_mode": _active_mode,
            "is_switching": _switching,
            "services": dict(zip(services, status, strict=True)),
        },
        status_code=200 if ready else 503,
    )


class SwitchModeRequest(BaseModel):
    mode: Literal["workspace", "gpu_worker", "hybrid"]


async def wait_for_services(client: httpx.AsyncClient, services: tuple[str, ...]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not all(is_process_running(service) for service in services):
            break
        if all(await asyncio.gather(*(is_service_healthy(client, s) for s in services))):
            return
        await asyncio.sleep(0.5)
    raise HTTPException(503, "Chế độ đích chưa sẵn sàng. Xem nhật ký dịch vụ.")


async def stop_idle_worker(client: httpx.AsyncClient) -> None:
    service = "gpu_worker"
    if not is_process_running(service):
        return
    token = os.environ["RA_ACCESS_TOKEN"]
    try:
        response = await client.post(
            f"http://{LOOPBACK}:{PORT_WORKER}/api/admin/worker/quiesce",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
    except httpx.RequestError:
        raise HTTPException(503, "Chưa xác nhận được GPU Worker đang rảnh.") from None
    if response.status_code == 409:
        raise HTTPException(409, "GPU Worker đang xử lý yêu cầu.")
    if response.status_code != 200:
        raise HTTPException(503, "Chưa xác nhận được GPU Worker đang rảnh.")
    await asyncio.to_thread(stop_process, service)


@app.post("/api/admin/switch_mode")
async def switch_mode(req: SwitchModeRequest, request: Request, _admin: Admin) -> dict[str, str]:
    global _active_mode, _switching
    if _switching:
        raise HTTPException(409, "Đang trong quá trình chuyển đổi chế độ.")
    target = MODE_SERVICES[req.mode]
    if _gpu_requests and "gpu_worker" not in target:
        raise HTTPException(409, "GPU Worker đang xử lý yêu cầu.")

    _switching = True
    previous = MODE_SERVICES[_active_mode]
    started = []
    client = request.app.state.upstream
    try:
        for service in target:
            if not is_process_running(service):
                spawn_service(service)
                started.append(service)
        await wait_for_services(client, target)
        for service in previous:
            if service not in target:
                if service == "gpu_worker":
                    await stop_idle_worker(client)
                else:
                    await asyncio.to_thread(stop_process, service)
        _active_mode = req.mode
        return {"status": "success", "active_mode": _active_mode}
    except (OSError, HTTPException, subprocess.SubprocessError) as exc:
        for service in started:
            if service not in previous:
                await asyncio.to_thread(stop_process, service)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(503, "Không thể chuyển chế độ. Xem nhật ký dịch vụ.") from None
    finally:
        _switching = False


@app.post("/api/admin/worker/reset")
async def reset_worker(request: Request, _admin: Admin) -> dict[str, str]:
    global _switching
    if _switching or _gpu_requests:
        raise HTTPException(409, "GPU Worker đang xử lý yêu cầu hoặc chuyển chế độ.")
    if "gpu_worker" not in MODE_SERVICES[_active_mode]:
        raise HTTPException(503, "GPU Worker chưa được bật trong chế độ hiện tại.")
    _switching = True
    try:
        client = request.app.state.upstream
        await stop_idle_worker(client)
        spawn_service("gpu_worker")
        await wait_for_services(client, ("gpu_worker",))
        return {"status": "ready", "active_mode": _active_mode}
    except (OSError, subprocess.SubprocessError):
        raise HTTPException(503, "Khởi động lại GPU Worker thất bại. Xem nhật ký.") from None
    finally:
        _switching = False


@app.get("/api/admin/logs")
def get_logs(
    _admin: Admin,
    service: Literal["workspace", "gpu_worker", "supervisor"] = "workspace",
    lines: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JSONResponse:
    path = RUN_DIR / LOG_NAMES[service]
    if path.is_symlink():
        raise HTTPException(404, "Không tìm thấy nhật ký dịch vụ.")
    try:
        with path.open("rb") as log:
            size = log.seek(0, os.SEEK_END)
            log.seek(max(0, size - LOG_TAIL_BYTES))
            data = log.read(LOG_TAIL_BYTES)
    except FileNotFoundError:
        data = b""
        size = 0
    except OSError:
        raise HTTPException(503, "Không thể đọc nhật ký dịch vụ.") from None
    if size > LOG_TAIL_BYTES:
        _, _, data = data.partition(b"\n")
    token = os.environ["RA_ACCESS_TOKEN"].encode()
    rows = data.replace(token, b"[redacted]").decode("utf-8", errors="replace").splitlines()
    return JSONResponse(
        {"service": service, "lines": min(lines, len(rows)), "logs": "\n".join(rows[-lines:])},
        headers={"Cache-Control": "no-store"},
    )


def proxy_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    connection_headers = {name.strip().lower() for name in headers.get("connection", "").split(",")}
    blocked = HOP_HEADERS | connection_headers
    return [(key, value) for key, value in headers.multi_items() if key not in blocked]


async def proxy(request: Request, port: int, path: str, *, gpu: bool = False) -> Response:
    headers = dict(proxy_headers(httpx.Headers(request.headers.items())))
    headers.pop("x-forwarded-host", None)
    headers["x-forwarded-proto"] = request.url.scheme
    if gpu:
        headers.pop("cookie", None)
    content: bytes | AsyncIterator[bytes] | None = None
    if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
        if gpu:
            data = bytearray()
            async for part in request.stream():
                if len(data) + len(part) > MAX_GPU_BODY:
                    raise HTTPException(413, "Yêu cầu GPU quá lớn.")
                data.extend(part)
            content = bytes(data)
            headers.pop("content-length", None)
        else:
            content = request.stream()
    else:
        headers.pop("content-length", None)
    try:
        upstream = await request.app.state.upstream.request(
            request.method,
            f"http://{LOOPBACK}:{port}/{path}",
            content=content,
            headers=headers,
            params=request.query_params,
        )
    except httpx.RequestError:
        raise HTTPException(503, "Dịch vụ đích chưa sẵn sàng.") from None
    response = Response(content=upstream.content, status_code=upstream.status_code)
    for key, value in proxy_headers(upstream.headers):
        if key not in {"content-length", "content-encoding"}:
            response.headers.append(key, value)
    return response


@app.get("/api/admin/vram")
@app.post("/api/admin/vram/clear")
async def worker_maintenance(request: Request, _admin: Admin) -> Response:
    global _gpu_requests
    if _switching or "gpu_worker" not in MODE_SERVICES[_active_mode]:
        raise HTTPException(503, "GPU Worker chưa sẵn sàng.")
    _gpu_requests += 1
    try:
        path = request.url.path.removeprefix("/")
        return await proxy(request, PORT_WORKER, path, gpu=True)
    finally:
        _gpu_requests -= 1


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def reverse_proxy(request: Request, path: str) -> Response:
    global _gpu_requests
    if path == "api/admin" or path.startswith("api/admin/"):
        require_bearer(request)
        raise HTTPException(404, "Không tìm thấy thao tác quản trị.")
    if path in GPU_ROUTES:
        require_bearer(request)
        if _switching or "gpu_worker" not in MODE_SERVICES[_active_mode]:
            raise HTTPException(503, "GPU Worker chưa sẵn sàng.")
        _gpu_requests += 1
        try:
            return await proxy(request, PORT_WORKER, path, gpu=True)
        finally:
            _gpu_requests -= 1
    if "workspace" not in MODE_SERVICES[_active_mode]:
        raise HTTPException(503, "Workspace chưa sẵn sàng.")
    return await proxy(request, PORT_WORKSPACE, path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=LOOPBACK, port=8000)
