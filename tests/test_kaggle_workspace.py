import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_assist.api.app import create_app
from research_assist.config import Settings
from scripts.package_source import package_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook_runtime():
    notebook = json.loads(
        (ROOT / "notebooks/04_kaggle_workspace.ipynb").read_text(encoding="utf-8")
    )
    namespace = {"__name__": "kaggle_workspace"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert not namespace["START_WORKSPACE"] and not namespace["OPEN_PUBLIC_TUNNEL"]
    return namespace


def test_workspace_source_archive_and_extraction(notebook_runtime, tmp_path):
    archive = tmp_path / "ResearchDesk_Source.zip"
    package_source(ROOT, archive)
    with zipfile.ZipFile(archive) as zipped:
        files = zipped.namelist()
        assert "research-assist/SOURCE_MANIFEST.json" in files
        assert "research-assist/notebooks/boot.sh" in files
        assert "research-assist/.env" not in files
        assert "research-assist/notebooks/kaggle_bundle.zip" not in files
    extracted = notebook_runtime["unpack_workspace"](archive, tmp_path / "app")
    assert (extracted / "pyproject.toml").read_bytes() == (ROOT / "pyproject.toml").read_bytes()
    assert (extracted / "src/research_assist/web/index.html").is_file()
    launcher = extracted / "notebooks/boot.sh"
    assert launcher.read_bytes() == (ROOT / "notebooks/boot.sh").read_bytes()
    assert b"\r" not in launcher.read_bytes()
    command = notebook_runtime["workspace_command"]
    assert command(extracted, "/venv/bin/python", 8000) == [
        "bash",
        str(launcher),
        "/venv/bin/python",
        "8000",
    ]
    assert command(tmp_path, "/venv/bin/python", 8000) == [
        "/venv/bin/python",
        "-m",
        "research_assist.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert not (extracted / ".env.example").exists()
    malicious = tmp_path / "bad.zip"
    with zipfile.ZipFile(malicious, "w") as zipped:
        zipped.writestr("research-assist/src/research_assist/../../../escaped.py", "")
    with pytest.raises(ValueError, match="Invalid path"):
        notebook_runtime["unpack_workspace"](malicious, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_workspace_tunnel_hostname_login_and_assets(notebook_runtime, tmp_path, monkeypatch):
    env = notebook_runtime["workspace_environment"](
        tmp_path,
        "sample-token-for-test-only-32-bytes",
        "https://preview.trycloudflare.com",
        openalex_key="example-openalex-key",
    )
    assert env["RA_OPENALEX_API_KEY"] == "example-openalex-key"
    for name, value in env.items():
        if name.startswith("RA_"):
            monkeypatch.setenv(name, value)
    app = create_app(Settings(_env_file=None))
    with TestClient(app, base_url="https://preview.trycloudflare.com") as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/documents").status_code == 401
        login = client.post(
            "/api/auth/login",
            json={"token": env["RA_ACCESS_TOKEN"]},
            headers={"Origin": "https://preview.trycloudflare.com"},
        )
        assert login.status_code == 200 and "Secure" in login.headers["Set-Cookie"]
        assert client.get("/api/documents").status_code == 200
        assert client.get("https://untrusted.example/api/health").status_code == 400


def test_kaggle_deployment_package(tmp_path, monkeypatch):
    archive = tmp_path / "ResearchDesk_Source.zip"
    package_source(ROOT, archive)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        expected = {
            "research-assist/kaggle/boot.sh",
            "research-assist/kaggle/__init__.py",
            "research-assist/kaggle/kaggle_master_worker.ipynb",
            "research-assist/kaggle/kaggle_server.py",
            "research-assist/kaggle/kaggle_supervisor_gateway.py",
            "research-assist/kaggle/run_kaggle.py",
            "research-assist/kaggle/README.md",
            "research-assist/kaggle/security.py",
        }
        assert expected.issubset(names)

    boot_sh = (ROOT / "kaggle/boot.sh").read_bytes()
    assert b"\r" not in boot_sh
    assert b"-m kaggle.run_kaggle" in boot_sh

    nb_data = json.loads((ROOT / "kaggle/kaggle_master_worker.ipynb").read_text(encoding="utf-8"))
    assert (nb_data["nbformat"], nb_data["nbformat_minor"]) == (4, 5)
    code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
    (cell,) = code_cells
    assert cell["execution_count"] is None and cell["outputs"] == []

    root = tmp_path / "extracted"
    root.mkdir()

    def staged_source(directory, pattern):
        assert directory == Path("/kaggle/input") and pattern == archive.name
        return iter([archive])

    calls = []

    def run_launcher(command, *, cwd, env, check):
        assert cwd == root and env["PROJECT_DIR"] == str(root) and check
        calls.append(command)

    monkeypatch.setattr(Path, "rglob", staged_source)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **_: str(root))
    monkeypatch.setattr(subprocess, "run", run_launcher)
    exec(compile("".join(cell["source"]), cell["id"], "exec"), {})
    assert calls == [["bash", "kaggle/boot.sh", "--mode", "workspace"]]
    assert (root / "kaggle/boot.sh").read_bytes() == boot_sh

