"""Build standalone train+serve and inference notebooks without changing older notebooks."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


INTRO = """
# E5-base fine-tune + GPU endpoint cho Research-assist

Chọn **intfloat/multilingual-e5-base** cho lượt đầu: encoder đa ngôn ngữ, prefix
`query:` / `passage:` nhất quán, pipeline CachedMNRL đã có sẵn trong project.
Đây là lựa chọn thực dụng về tích hợp/huấn luyện, không phải tuyên bố model tốt
nhất mọi dataset. Qwen3-Embedding-0.6B là đối chứng sau khi có baseline riêng.

Notebook train dùng một GPU T4, không cộng VRAM của hai T4. QASPER mặc định là
tiếng Anh; bổ sung dữ liệu Việt–Anh đã review để đo và cải thiện đúng miền sử dụng.
Các model LLM planning/writing vẫn dùng cấu hình riêng của app.

Các cell cuối **tùy chọn** chạy API GPU sau khi train. Cloudflare chỉ chuyển tiếp
HTTP; GPU đến từ máy đang chạy notebook. URL tạm hết hiệu lực khi tunnel dừng.
Không coi phiên notebook là dịch vụ 24/7. Với Kaggle, kiểm tra điều khoản hiện hành
và chỉ dùng endpoint nếu hình thức sử dụng này được nền tảng cho phép; không có
keep-alive hay cơ chế vượt quota. Có thể dùng phần serve trên GPU host cho phép.

Nếu train gần hết giờ phiên, tải checkpoint ZIP, mở notebook 04 trong phiên mới,
thêm ZIP làm Input rồi chỉ chạy inference. Không train và serve cùng lúc.

Nguồn chính thức:
- https://huggingface.co/intfloat/multilingual-e5-base
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- https://www.kaggle.com/docs/notebooks
- https://www.kaggle.com/terms
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
"""

SETUP = r"""
import os, sys, subprocess
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
_RESEARCHDESK_ENV_READY = False
status = install_training_environment("/kaggle/temp/researchdesk-inference-environment.json")
_RESEARCHDESK_ENV_READY = status["ready"]
if _RESEARCHDESK_ENV_READY:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.27,<1"])
    print("Environment ready. Run the checkpoint selection cell.")
"""

CONFIG = r"""
# Train+serve: MODEL_OUT is set by training above. In a fresh notebook, use ZIP or path.
CHECKPOINT_PATH = str(globals().get("MODEL_OUT", ""))
CHECKPOINT_ZIP = ""  # e.g. /kaggle/input/my-e5/research-e5-checkpoint.zip
PORT = 8008
MAX_SEQ_LENGTH = 512
START_GPU_SERVER = False  # Run All can train/export without waiting for an API key
START_CLOUDFLARE = False  # opt in only on a host/platform permitting remote inference
API_KEY_SECRET_NAME = "RESEARCH_GPU_API_KEY"  # create this Kaggle Secret, >=32 chars
"""

LOAD = r"""
if globals().get("_RESEARCHDESK_ENV_READY", False) and START_GPU_SERVER:
    import gc, getpass, os, threading, time
    import torch, uvicorn, httpx
    from IPython.display import display, Markdown
    if "GPU_SERVER" in globals():
        raise RuntimeError("Stop the previous server using the cleanup cell before starting again.")
    # If training ran in this kernel, release its model/optimizer before serving.
    gc.collect()
    torch.cuda.empty_cache()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator before starting this endpoint.")
    if CHECKPOINT_ZIP:
        selected_path = extract_checkpoint(CHECKPOINT_ZIP, "/kaggle/temp/research-e5-serve")
    elif CHECKPOINT_PATH and Path(CHECKPOINT_PATH).is_dir():
        selected_path = Path(CHECKPOINT_PATH)
    else:
        found = list(Path("/kaggle/input").rglob("research-e5-checkpoint.zip"))
        if len(found) != 1:
            raise RuntimeError("Set CHECKPOINT_ZIP or CHECKPOINT_PATH to the trained checkpoint.")
        selected_path = extract_checkpoint(found[0], "/kaggle/temp/research-e5-serve")
    try:
        from kaggle_secrets import UserSecretsClient
        API_KEY = UserSecretsClient().get_secret(API_KEY_SECRET_NAME)
    except Exception:
        API_KEY = getpass.getpass("Enter your private GPU API key (at least 32 characters): ")
    if len(API_KEY) < 32:
        raise ValueError("Use at least 32 characters; put the same key in your local app config.")
    GPU_ENCODER, GPU_MODEL_ID = load_e5_checkpoint(selected_path, "cuda", MAX_SEQ_LENGTH)
    GPU_APP = create_embedding_app(GPU_ENCODER, API_KEY, GPU_MODEL_ID, MAX_SEQ_LENGTH)
    GPU_SERVER = uvicorn.Server(uvicorn.Config(GPU_APP, host="127.0.0.1", port=PORT,
                               log_level="warning", access_log=False))
    GPU_THREAD = threading.Thread(target=GPU_SERVER.run, daemon=True)
    GPU_THREAD.start()
    deadline = time.monotonic() + 60
    while not GPU_SERVER.started and GPU_THREAD.is_alive() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not GPU_SERVER.started:
        raise RuntimeError("Server did not start; inspect the private notebook log.")
    local_url = f"http://127.0.0.1:{PORT}"
    check = httpx.post(local_url + "/v1/embeddings", headers={"Authorization": "Bearer " + API_KEY},
        json={"model": GPU_MODEL_ID, "input": ["How is evidence retrieved?"],
              "input_type": "query"}, timeout=60)
    check.raise_for_status()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model ID:", GPU_MODEL_ID)
    print("Embedding dimension:", len(check.json()["data"][0]["embedding"]))
    print("Local API test passed. The API key is intentionally not printed/saved.")
elif globals().get("_RESEARCHDESK_ENV_READY", False):
    print("Server disabled. Set START_GPU_SERVER=True when you want to serve the checkpoint.")
"""

TUNNEL = r"""
if globals().get("_RESEARCHDESK_ENV_READY", False) and START_CLOUDFLARE:
    import hashlib, json, re, shutil, subprocess, threading, time, urllib.request
    if "GPU_SERVER" not in globals() or not GPU_SERVER.started:
        raise RuntimeError("Start the GPU server and pass its local API check before tunneling.")
    if "CLOUDFLARE_PROCESS" in globals() and CLOUDFLARE_PROCESS.poll() is None:
        raise RuntimeError("Stop the existing tunnel before creating another one.")
    binary = shutil.which("cloudflared")
    if not binary:
        # Resolve an official release and verify GitHub's published asset digest.
        req = urllib.request.Request("https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
                                     headers={"User-Agent": "Research-assist-notebook"})
        with urllib.request.urlopen(req, timeout=30) as response:
            release = json.load(response)
        asset = next(a for a in release["assets"] if a["name"] == "cloudflared-linux-amd64")
        digest = asset.get("digest", "")
        if not digest.startswith("sha256:"):
            raise RuntimeError("No published SHA256 digest. Install verified cloudflared manually.")
        binary = "/kaggle/temp/cloudflared"
        urllib.request.urlretrieve(asset["browser_download_url"], binary)
        with open(binary, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest.split(":", 1)[1]:
            raise RuntimeError("cloudflared checksum mismatch; refusing execution.")
        os.chmod(binary, 0o700)
    CLOUDFLARE_PROCESS = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--url",
        f"http://127.0.0.1:{PORT}"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    TUNNEL_URL = None
    TUNNEL_READY = threading.Event()
    def read_tunnel_log():
        global TUNNEL_URL
        for line in CLOUDFLARE_PROCESS.stderr:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                TUNNEL_URL = match.group(0)
                TUNNEL_READY.set()
    threading.Thread(target=read_tunnel_log, daemon=True).start()
    if not TUNNEL_READY.wait(60):
        CLOUDFLARE_PROCESS.terminate()
        raise RuntimeError("No tunnel URL received. Check network/platform permissions.")
    print("Copy this URL:", TUNNEL_URL)
    print("\nLocal app .env (replace existing entries):")
    print("RA_EMBEDDING_PROVIDER=remote")
    print("RA_EMBEDDING_BASE_URL=" + TUNNEL_URL + "/v1")
    print("RA_EMBEDDING_MODEL=" + GPU_MODEL_ID)
    print("RA_EMBEDDING_API_KEY=<the same private key you entered above>")
    print("RA_EMBEDDING_MAX_SEQ_LENGTH=" + str(MAX_SEQ_LENGTH))
    print("RA_EMBEDDING_BATCH_SIZE=16")
    print("RA_CHUNK_STRATEGY=tokens")
    print("RA_CHUNK_TOKENS=384")
    print("RA_CHUNK_TOKEN_OVERLAP=48")
    print("RA_CHUNK_TOKENIZER_MODEL=intfloat/multilingual-e5-base")
    print("Restart the app. A new checkpoint requires Reindex.")
    print("A new URL for the same model ID does not require Reindex.")
elif globals().get("_RESEARCHDESK_ENV_READY", False):
    print("Tunnel disabled. Set START_CLOUDFLARE=True only if your host permits it.")
"""

CLEANUP = r"""
# Run explicitly when you finish. This cell is not part of Run All.
STOP_ENDPOINT = False
if STOP_ENDPOINT:
    if "CLOUDFLARE_PROCESS" in globals() and CLOUDFLARE_PROCESS.poll() is None:
        CLOUDFLARE_PROCESS.terminate()
        CLOUDFLARE_PROCESS.wait(timeout=15)
    if "GPU_SERVER" in globals():
        GPU_SERVER.should_exit = True
        GPU_THREAD.join(timeout=60)
        if GPU_THREAD.is_alive():
            raise RuntimeError("GPU work is still draining. Wait before unloading the model.")
        del GPU_SERVER, GPU_APP, GPU_ENCODER
        gc.collect()
        torch.cuda.empty_cache()
    print("Endpoint stopped.")
"""


def serve_cells():
    server = (ROOT / "notebooks/gpu_embedding_server.py").read_text(encoding="utf-8")
    return [
        cell(
            "markdown",
            "## GPU inference và Cloudflare Tunnel (tùy chọn)\n"
            "Giữ notebook ở chế độ tương tác; không commit/save output có secret. "
            "Checkpoint và model ID được giữ riêng với URL tạm. "
            "Kaggle không phải hosting luôn bật; "
            "kiểm tra quyền sử dụng nền tảng trước khi bật tunnel.",
        ),
        cell(
            "code",
            'if globals().get("_RESEARCHDESK_ENV_READY", False):\n'
            "    import subprocess, sys\n"
            '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", '
            '"fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.27,<1"])',
        ),
        cell(
            "code",
            'if globals().get("_RESEARCHDESK_ENV_READY", False):\n'
            + textwrap.indent(server, "    "),
        ),
        cell("code", CONFIG),
        cell("code", LOAD),
        cell("code", TUNNEL),
        cell("code", CLEANUP),
    ]


def write(name, cells):
    for index, value in enumerate(cells):
        value["id"] = f"gpu-e5-{index:03d}"
        if value["cell_type"] == "code":
            value.update(execution_count=None, outputs=[])
    book = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
    }
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def build():
    original = json.loads((ROOT / "notebooks/01_kaggle_finetune_e5.ipynb").read_text("utf-8"))
    write(
        "03_kaggle_e5_train_and_serve.ipynb",
        [cell("markdown", INTRO)] + original["cells"] + serve_cells(),
    )
    environment = (ROOT / "notebooks/kaggle_environment.py").read_text("utf-8")
    write(
        "04_gpu_e5_inference_cloudflare.ipynb",
        [
            cell(
                "markdown",
                INTRO
                + "\n## Phiên inference riêng\nThêm checkpoint ZIP từ notebook 01/03 làm Input. "
                "Notebook này không train lại model.\n",
            ),
            cell("code", environment + "\n" + SETUP),
        ]
        + serve_cells(),
    )


if __name__ == "__main__":
    build()
