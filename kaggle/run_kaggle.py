"""Launch the Kaggle supervisor and optionally expose it through a Quick Tunnel."""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from kaggle.processes import stop_owned_process as stop_process
from notebooks.kaggle_tunnel import start_quick_tunnel
from notebooks.kaggle_workspace import check_loopback_port

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODES = ("workspace", "gpu_worker", "hybrid")
PORT = 8000


def secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name).strip()
    except Exception:
        return ""


def install_dependencies() -> None:
    dependencies = [
        "fastapi",
        "uvicorn",
        "pydantic_settings",
        "httpx",
        "multipart",
        "pymupdf",
        "docx",
        "numpy",
        "sentence_transformers",
    ]
    if any(importlib.util.find_spec(module) is None for module in dependencies):
        package = f"{PROJECT_DIR}[local-models]"
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def wait_for_gateway(process: subprocess.Popen, base_url: str, token: str, mode: str) -> None:
    import httpx

    auth = {"Authorization": f"Bearer {token}"}
    checks = ["/api/admin/logs?lines=1"]
    if mode != "gpu_worker":
        checks.append("/api/documents")
    if mode != "workspace":
        checks.append("/v1/models")

    deadline = time.monotonic() + 90
    with httpx.Client(base_url=base_url, timeout=5, trust_env=False) as client:
        while process.poll() is None and time.monotonic() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    for path in checks:
                        if client.get(path).status_code != 401:
                            raise RuntimeError("Gateway không vượt qua kiểm tra xác thực.")
                        if client.get(path, headers=auth).status_code != 200:
                            break
                    else:
                        return
            except httpx.RequestError:
                pass
            time.sleep(0.5)
    raise RuntimeError("Gateway chưa sẵn sàng. Xem supervisor.log trong thư mục nhật ký.")


def run(mode: str, public_tunnel: bool) -> None:
    token = secret("RA_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Thêm RA_ACCESS_TOKEN hợp lệ vào Kaggle Secrets (32–256 ký tự URL-safe)."
        )
    check_loopback_port(PORT)
    install_dependencies()
    from kaggle.security import LOOPBACK, valid_access_token

    if not valid_access_token(token):
        raise RuntimeError("RA_ACCESS_TOKEN phải có 32–256 ký tự URL-safe.")

    run_dir = Path(os.environ.get("RA_KAGGLE_RUN_DIR", "/kaggle/temp/research-assist-master"))
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "RA_ACCESS_TOKEN": token,
        "INITIAL_MODE": mode,
        "RA_KAGGLE_RUN_DIR": str(run_dir),
        "RA_DATA_DIR": os.environ.get("RA_DATA_DIR", str(run_dir / "data")),
        "PYTHONPATH": os.pathsep.join((str(PROJECT_DIR / "src"), str(PROJECT_DIR))),
    }
    for name in ("RA_GEMINI_API_KEY", "RA_OPENALEX_API_KEY"):
        value = secret(name)
        if value:
            env[name] = value
    if env.get("RA_GEMINI_API_KEY") and "RA_LLM_PROVIDER" not in env:
        env["RA_LLM_PROVIDER"] = "gemini"
    with (run_dir / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "kaggle.kaggle_supervisor_gateway:app",
                "--host",
                LOOPBACK,
                "--port",
                str(PORT),
            ],
            cwd=PROJECT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    url = f"http://{LOOPBACK}:{PORT}"
    tunnel = None
    try:
        wait_for_gateway(process, url, token, mode)
        if public_tunnel:
            tunnel, url = start_quick_tunnel(PORT, run_dir)
            wait_for_gateway(process, url, token, mode)

        print(f"Gateway đã sẵn sàng ở chế độ {mode}.")
        print(f"WORKSPACE_URL={url}")
        print("Dùng RA_ACCESS_TOKEN từ Kaggle Secrets hoặc biến môi trường để đăng nhập.")
        if mode != "workspace":
            print(f"GPU embedding base URL: {url}/v1")
            print(f"Tra cứu model ID, tokenizer và giới hạn token: GET {url}/v1/info (Bearer).")
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if tunnel is not None:
                stop_process(tunnel)
        finally:
            stop_process(process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=MODES, default=os.environ.get("INITIAL_MODE", "workspace")
    )
    parser.add_argument("--public-tunnel", action="store_true")
    options = parser.parse_args()
    if options.mode not in MODES:
        parser.error("INITIAL_MODE phải là workspace, gpu_worker hoặc hybrid.")
    run(options.mode, options.public_tunnel)
