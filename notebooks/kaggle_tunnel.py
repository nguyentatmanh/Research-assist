"""Start and stop an optional Cloudflare Quick Tunnel in an interactive notebook."""

import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def find_cloudflared(working_dir: Path | str) -> str:
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise RuntimeError("Install cloudflared for your operating system and put it on PATH.")
    arch = "amd64" if machine in {"x86_64", "amd64"} else "arm64"
    name = f"cloudflared-linux-{arch}"
    directory = Path(working_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
        headers={"User-Agent": "Research-Assist-Notebook"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        release = json.load(response)
    asset = next((item for item in release["assets"] if item["name"] == name), None)
    if not asset or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", asset.get("digest") or ""):
        raise RuntimeError("No published SHA256 digest for cloudflared. Install it manually.")
    digest = asset["digest"].split(":", 1)[1].lower()
    target = directory / name
    if target.is_file():
        with target.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == digest:
                target.chmod(0o700)
                return str(target.resolve())
    deadline = time.monotonic() + 90
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            with urllib.request.urlopen(asset["browser_download_url"], timeout=15) as response:
                while block := response.read(1024 * 1024):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Downloading cloudflared took too long.")
                    checksum.update(block)
                    output.write(block)
        if checksum.hexdigest() != digest:
            raise RuntimeError("cloudflared does not match the official release checksum.")
        temporary.chmod(0o700)
        os.replace(temporary, target)
        return str(target.resolve())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stop_process(process: subprocess.Popen) -> None:
    """Reap the process even if it ignores a graceful stop."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_quick_tunnel(
    port: int, working_dir: Path | str, *, timeout: float = 60
) -> tuple[subprocess.Popen, str]:
    """Return the owned process and assigned HTTPS URL, or tear it down on error."""
    if not 1 <= port <= 65535 or timeout <= 0:
        raise ValueError("Invalid tunnel port or timeout.")
    executable = find_cloudflared(working_dir)
    process = subprocess.Popen(
        [executable, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    results: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        stream = process.stdout
        if stream is None:
            results.put(None)
            return
        found = False
        with stream:
            for line in stream:
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com\b", line)
                if not found and match:
                    found = True
                    results.put(match.group())
        results.put(None)

    try:
        threading.Thread(target=read_output, daemon=True).start()
        try:
            url = results.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No Cloudflare URL received before the timeout.") from None
        if not url or process.poll() is not None:
            raise RuntimeError("cloudflared exited before the tunnel was ready.")
        return process, url
    except BaseException:
        stop_process(process)
        raise
