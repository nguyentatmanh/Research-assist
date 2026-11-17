"""Generate the Research Assist full workspace notebook."""

import json
from pathlib import Path

from build_gpu_notebooks import ROOT, cell

INTRO = """
# Chạy toàn bộ Research Assist trên Kaggle

Notebook này cài và chạy chính backend + giao diện Research Assist.
Tải notebook và gói mã nguồn từ cùng một bản. Trên máy dự án, tạo ZIP bằng:

```powershell
.\.venv\Scripts\python.exe scripts/package_source.py
```

Đưa `dist/ResearchDesk_Source.zip` vào một Kaggle Dataset **riêng tư** rồi chọn **Add Input**.
Notebook lấy `notebooks/boot.sh` từ ZIP để khởi động backend.
Bật Internet để cài thư viện. Tạo và bật Kaggle Secret `RA_ACCESS_TOKEN` gồm 32–256 ký tự
URL-safe ngẫu nhiên (chữ, số, `_`, `-`). Dùng secret này để đăng nhập web.
Nếu chọn Gemini, cần thêm `RA_GEMINI_API_KEY`.
Literature Scout có thể tìm OpenAlex ngay. Để dùng hạn mức theo tài khoản, thêm
`RA_OPENALEX_API_KEY` vào Kaggle Secrets và bật `USE_OPENALEX_KEY`.
GPU chỉ cần thiết khi chọn embedding/reranker CUDA. Mặc định Demo giúp thử luồng
tải tài liệu, đặt câu hỏi và mở trích dẫn mà không gọi API mô hình.

Chạy các cell theo thứ tự trong phiên tương tác. Nếu được phép mở Cloudflare Quick Tunnel,
đặt cả `START_WORKSPACE` và `OPEN_PUBLIC_TUNNEL` thành `True`, rồi mở URL ở cell khởi động.
Dừng bằng cell cuối. URL và tiến trình sống cùng phiên Kaggle.
SQLite, tài liệu và cấu hình riêng nằm trong `/kaggle/temp/research-assist-data` trong phiên này.
"""

CONFIG = r"""
from pathlib import Path

SOURCE_ZIP = ""  # Trống: tự tìm đúng một ResearchDesk_Source.zip trong Add Input.
DATA_DIR = Path("/kaggle/temp/research-assist-data")  # Riêng tư, không đưa vào Kaggle Output.
PORT = 8000
START_WORKSPACE = False
OPEN_PUBLIC_TUNNEL = False
USE_GEMINI = False
GEMINI_MODEL = ""  # Trống: dùng model mặc định trong ứng dụng.
USE_OPENALEX_KEY = False
USE_GPU_RETRIEVAL = False
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Hub ID hoặc thư mục model tương thích.
RERANKER_MODEL = ""  # Tùy chọn CrossEncoder tương thích, dùng cùng GPU.
"""

INSTALL = r"""
if START_WORKSPACE:
    import sys
    from uuid import uuid4

    if sys.version_info < (3, 11):
        raise RuntimeError("Research Assist requires Python 3.11 or newer.")
    access_token = kaggle_secret("RA_ACCESS_TOKEN", min_length=32)
    gemini_key = kaggle_secret("RA_GEMINI_API_KEY") if USE_GEMINI else ""
    openalex_key = kaggle_secret("RA_OPENALEX_API_KEY") if USE_OPENALEX_KEY else ""
    if USE_GPU_RETRIEVAL:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU accelerator before selecting CUDA retrieval.")
    if SOURCE_ZIP:
        source_zip = Path(SOURCE_ZIP)
    else:
        matches = sorted(Path("/kaggle/input").rglob("ResearchDesk_Source.zip"))
        if len(matches) != 1:
            raise RuntimeError("Add one ResearchDesk_Source.zip as Input or set SOURCE_ZIP.")
        (source_zip,) = matches
    work_root = Path("/kaggle/temp/research-assist-runtime")
    work_root.mkdir(parents=True, exist_ok=True)
    source = unpack_workspace(source_zip, work_root / ("source-" + uuid4().hex))
    env_dir = work_root / "venv"
    # Isolate app packages while reusing the Kaggle image's CUDA PyTorch.
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(env_dir)],
        check=True,
        timeout=180,
    )
    workspace_python = str(env_dir / "bin/python")
    package = str(source) + ("[local-models]" if USE_GPU_RETRIEVAL else "")
    subprocess.run(
        [workspace_python, "-m", "pip", "install", package],
        cwd=work_root,
        check=True,
        timeout=900,
    )
else:
    print("Đặt START_WORKSPACE=True trong cell cấu hình khi sẵn sàng.")
"""

START = r"""
if START_WORKSPACE:
    if "workspace_python" not in globals():
        raise RuntimeError("Run the source installation cell first.")
    if "workspace_process" in globals() and workspace_process.poll() is None:
        raise RuntimeError("Stop the current workspace using the last cell before restarting.")
    if "workspace_tunnel" in globals() and workspace_tunnel and workspace_tunnel.poll() is None:
        raise RuntimeError("Stop the current tunnel using the last cell before restarting.")
    check_loopback_port(PORT)
    local_url = f"http://127.0.0.1:{PORT}"
    public_url = None
    workspace_tunnel = workspace_process = None
    try:
        if OPEN_PUBLIC_TUNNEL:
            workspace_tunnel, public_url = start_quick_tunnel(PORT, work_root)
        env = workspace_environment(
            DATA_DIR, access_token, public_url, gemini_key=gemini_key,
            gemini_model=GEMINI_MODEL, openalex_key=openalex_key, use_gpu=USE_GPU_RETRIEVAL,
            embedding_model=EMBEDDING_MODEL, reranker_model=RERANKER_MODEL,
        )
        log_path = work_root / "workspace.log"
        with log_path.open("w", encoding="utf-8") as log:
            workspace_process = subprocess.Popen(
                workspace_command(source, workspace_python, PORT),
                cwd=work_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print("Đang kiểm tra ứng dụng và xác thực. Log:", log_path)
        wait_for_workspace(workspace_process, local_url, access_token)
        if public_url:
            wait_for_workspace(workspace_process, public_url, access_token, timeout=90)
            print("Mở Research Assist:", public_url)
            print("Đăng nhập bằng giá trị RA_ACCESS_TOKEN từ Kaggle Secrets.")
        else:
            print("Workspace hoạt động trong phiên Kaggle:", local_url)
            print("Để truy cập từ trình duyệt bên ngoài, dừng rồi bật OPEN_PUBLIC_TUNNEL.")
    except BaseException:
        try:
            if workspace_tunnel:
                stop_process(workspace_tunnel)
        finally:
            if workspace_process:
                stop_process(workspace_process)
        raise
"""

STOP = r"""
# Chỉ đổi True rồi chạy riêng cell này lúc đã dùng xong.
STOP_WORKSPACE = False
if STOP_WORKSPACE:
    try:
        if globals().get("workspace_tunnel"):
            stop_process(workspace_tunnel)
    finally:
        if globals().get("workspace_process"):
            stop_process(workspace_process)
    print("Đã dừng workspace và tunnel.")
"""


def build() -> Path:
    cells = [
        cell("markdown", INTRO),
        cell("code", CONFIG),
        cell("code", (ROOT / "notebooks/kaggle_tunnel.py").read_text(encoding="utf-8")),
        cell("code", (ROOT / "notebooks/kaggle_workspace.py").read_text(encoding="utf-8")),
        cell("markdown", "## Nạp và cài mã nguồn"),
        cell("code", INSTALL),
        cell("markdown", "## Khởi động và lấy URL"),
        cell("code", START),
        cell("markdown", "## Dừng khi dùng xong"),
        cell("code", STOP),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"workspace-{index:02d}"
    book = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    path = ROOT / "notebooks/04_kaggle_workspace.ipynb"
    path.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())
