"""Shared runtime of the Research Assist workspace notebook (stdlib only)."""

import http.client
import json
import os
import re
import socket
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

WORKSPACE_BOOT = Path("notebooks/boot.sh")


def unpack_workspace(archive_path: Path, destination: Path) -> Path:
    """Unpack a source-only archive into a new directory."""
    if destination.exists():
        raise FileExistsError("Choose a new directory for the source archive.")
    with zipfile.ZipFile(archive_path) as archive:
        files = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if parts and parts[0] == "research-assist":
                parts = parts[1:]
            if (
                not parts
                or member.filename.startswith("/")
                or any(part in {"..", "."} or "\\" in part or ":" in part for part in parts)
            ):
                raise ValueError("Invalid path in the workspace archive.")
            if parts not in {("pyproject.toml",), ("LICENSE",), WORKSPACE_BOOT.parts} and (
                parts[:2] != ("src", "research_assist")
                or any(part.startswith(".") or part == "__pycache__" for part in parts)
            ):
                continue
            path = destination.joinpath(*parts)
            if path in files:
                raise ValueError("Duplicate file in workspace archive.")
            files[path] = member
        required = {destination / "pyproject.toml", destination / "src/research_assist/cli.py"}
        if not required.issubset(files):
            raise ValueError("Archive does not contain Research Assist source.")
        if sum(member.file_size for member in files.values()) > 100 * 1024**2:
            raise ValueError("Workspace source archive exceeds 100 MiB when unpacked.")
        for path, member in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
    return destination


def workspace_command(source: Path, python: str, port: int) -> list[str]:
    launcher = source / WORKSPACE_BOOT
    if launcher.is_file():
        return ["bash", str(launcher), python, str(port)]
    return [
        python,
        "-m",
        "research_assist.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def kaggle_secret(name: str, *, min_length: int = 1) -> str:
    from kaggle_secrets import UserSecretsClient

    try:
        value = UserSecretsClient().get_secret(name).strip()
    except Exception:
        raise ValueError(f"Add and enable the Kaggle Secret {name}.") from None
    if len(value) < min_length or any(c in value for c in "\r\n"):
        raise ValueError(f"{name} must be a single line of at least {min_length} characters.")
    return value


def workspace_environment(
    data_dir: Path,
    access_token: str,
    public_url: str | None,
    *,
    gemini_key: str = "",
    gemini_model: str = "",
    openalex_key: str = "",
    use_gpu: bool = False,
    embedding_model: str = "intfloat/multilingual-e5-small",
    reranker_model: str = "",
) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", access_token):
        raise ValueError("RA_ACCESS_TOKEN must be a URL-safe token of 32–256 characters.")
    hosts = ["localhost", "127.0.0.1"]
    if public_url:
        if not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", public_url):
            raise ValueError("Expected a Cloudflare Quick Tunnel HTTPS URL.")
        hostname = urlsplit(public_url).hostname
        if hostname is not None:
            hosts.append(hostname)

    env = {name: value for name, value in os.environ.items() if not name.startswith("RA_")}
    env.pop("PYTHONPATH", None)
    env.update(
        RA_ACCESS_TOKEN=access_token,
        RA_ALLOWED_HOSTS=json.dumps(hosts),
        RA_DATA_DIR=str(data_dir),
        RA_LLM_PROVIDER="gemini" if gemini_key else "demo",
        RA_LLM_MODEL=gemini_model if gemini_key else "",
        RA_GEMINI_API_KEY=gemini_key,
        RA_OPENALEX_API_KEY=openalex_key,
        RA_EMBEDDING_PROVIDER="sentence-transformers" if use_gpu else "none",
        RA_EMBEDDING_MODEL=embedding_model,
        RA_LOCAL_DEVICE="cuda" if use_gpu else "cpu",
        RA_RERANKER_PROVIDER="local" if use_gpu and reranker_model else "none",
        RA_RERANKER_MODEL=reranker_model if use_gpu else "",
        TOKENIZERS_PARALLELISM="false",
    )
    return env


def workspace_check(url: str, path: str, token: str = "") -> int:
    """Probe an explicitly selected origin without following redirects."""
    target = urlsplit(url)
    connection_cls = (
        http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    )
    with_connection = connection_cls(target.hostname, target.port, timeout=5)
    try:
        headers = {"Authorization": "Bearer " + token} if token else {}
        with_connection.request("GET", path, headers=headers)
        response = with_connection.getresponse()
        return response.status
    finally:
        with_connection.close()


def wait_for_workspace(
    process: subprocess.Popen,
    base_url: str,
    token: str,
    *,
    timeout: float = 90,
) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if workspace_check(base_url, "/api/health") == 200:
                if workspace_check(base_url, "/api/documents") != 401:
                    raise RuntimeError("Workspace refused to start without authentication.")
                if workspace_check(base_url, "/api/documents", token) == 200:
                    return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    raise RuntimeError("Workspace did not pass its startup and authentication checks.")


def check_loopback_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(
                f"Port {port} is in use. Stop the old server or pick another port."
            ) from None
