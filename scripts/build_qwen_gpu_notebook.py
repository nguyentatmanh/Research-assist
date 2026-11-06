"""Build a standalone Qwen3 GPU inference notebook for the remote embedding UI."""

import json

from build_gpu_notebooks import CLEANUP, ROOT, TUNNEL, cell, write

INTRO = """# Qwen3 Embedding GPU endpoint cho Research-assist

Notebook này **chạy inference**. Fine-tune bằng notebook **06** trước.
Chọn Qwen/Qwen3-Embedding-0.6B
hoặc Qwen/Qwen3-Embedding-4B, hoặc trỏ tới thư mục checkpoint Qwen3 đã huấn luyện.
Với T4, thử 0.6B trước; 4B có thể hết VRAM tùy độ dài chuỗi và GPU còn trống.
Không cộng VRAM của hai T4 vào một tiến trình.

Endpoint sử dụng `encode_query` / `encode_document`, khác prefix của E5.
Notebook mặc định không mở server/tunnel. Chỉ bật tunnel nếu nền tảng cho phép;
URL tạm chỉ sống trong phiên chạy. Không lưu key trong output notebook.

Nguồn: https://huggingface.co/Qwen/Qwen3-Embedding-4B
"""

SETUP = """import os, sys, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sentence-transformers==5.1.2", "transformers==4.57.1",
    "fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.27,<1",
    "huggingface-hub==0.36.0", "peft==0.17.1", "bitsandbytes==0.48.2", "accelerate==1.11.0"])
"""

CONFIG = """# Select one model, or set LOCAL_CHECKPOINT to a Qwen3 Embedding directory.
BASE_MODEL = "Qwen/Qwen3-Embedding-4B"  # or Qwen/Qwen3-Embedding-0.6B
LOCAL_CHECKPOINT = ""  # extracted qwen3-adapter folder from notebook 06, or full checkpoint
ADAPTER_ZIP = ""  # e.g. /kaggle/input/my-training/qwen3-adapter.zip from notebook 06
MAX_SEQ_LENGTH = 512  # start conservatively on T4
PORT = 8008
START_GPU_SERVER = False
START_CLOUDFLARE = False
API_KEY_SECRET_NAME = "RESEARCH_GPU_API_KEY"
"""

LOAD = """if START_GPU_SERVER:
    import gc, getpass, threading, time
    import httpx, torch, uvicorn
    from pathlib import Path
    from huggingface_hub import snapshot_download
    if "GPU_SERVER" in globals():
        raise RuntimeError("Stop the previous server before starting another one.")
    if BASE_MODEL not in {"Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"}:
        raise ValueError("Choose a supported Qwen3 Embedding base model.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator first.")
    if ADAPTER_ZIP:
        selected = extract_adapter_archive(ADAPTER_ZIP, "/kaggle/temp/qwen-adapter-input")
    else:
        selected = (Path(LOCAL_CHECKPOINT) if LOCAL_CHECKPOINT
                    else Path(snapshot_download(BASE_MODEL)))
    manifest_path = selected / "research_model.json"
    if (manifest_path.is_file()
            and json.loads(manifest_path.read_text()).get("format") == "swift_qwen3_adapter_v1"):
        adapter_info = read_adapter_manifest(selected)
        BASE_MODEL = adapter_info["base_model"]
        MAX_SEQ_LENGTH = adapter_info["max_seq_length"]
    try:
        from kaggle_secrets import UserSecretsClient
        API_KEY = UserSecretsClient().get_secret(API_KEY_SECRET_NAME)
    except Exception:
        API_KEY = getpass.getpass("Private GPU API key (at least 32 characters): ")
    if len(API_KEY) < 32:
        raise ValueError("Use a key with at least 32 characters.")
    GPU_ENCODER, GPU_MODEL_ID = load_qwen_checkpoint(selected, "cuda", MAX_SEQ_LENGTH)
    GPU_APP = create_embedding_app(GPU_ENCODER, API_KEY, GPU_MODEL_ID,
                                   MAX_SEQ_LENGTH, family="qwen3", tokenizer_model=BASE_MODEL)
    GPU_SERVER = uvicorn.Server(uvicorn.Config(GPU_APP, host="127.0.0.1", port=PORT,
                               log_level="warning", access_log=False))
    GPU_THREAD = threading.Thread(target=GPU_SERVER.run, daemon=True)
    GPU_THREAD.start()
    deadline = time.monotonic() + 60
    while not GPU_SERVER.started and GPU_THREAD.is_alive() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not GPU_SERVER.started:
        raise RuntimeError("GPU server did not start; inspect private notebook log.")
    check = httpx.post(f"http://127.0.0.1:{PORT}/v1/embeddings",
        headers={"Authorization": "Bearer " + API_KEY},
        json={"model": GPU_MODEL_ID, "input": ["Find relevant research evidence"],
              "input_type": "query"}, timeout=120)
    check.raise_for_status()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model ID:", GPU_MODEL_ID)
    print("Dimension:", len(check.json()["data"][0]["embedding"]))
    print("Local API test passed. API key is not printed.")
else:
    print("Server disabled. Set START_GPU_SERVER=True when ready.")
"""


def build():
    server = (ROOT / "notebooks/gpu_embedding_server.py").read_text(encoding="utf-8")
    adapter = (ROOT / "notebooks/qwen_adapter_encoder.py").read_text(encoding="utf-8")
    tunnel = TUNNEL.replace('globals().get("_RESEARCHDESK_ENV_READY", False) and ', "")
    tunnel = tunnel.replace('elif globals().get("_RESEARCHDESK_ENV_READY", False):', "else:")
    tunnel = tunnel.replace(
        'print("RA_CHUNK_TOKENIZER_MODEL=intfloat/multilingual-e5-base")',
        'print("RA_CHUNK_TOKENIZER_MODEL=" + BASE_MODEL)',
    )
    cells = [
        cell("markdown", INTRO),
        cell("code", SETUP),
        cell("code", adapter),
        cell("code", server),
        cell("code", CONFIG),
        cell("code", LOAD),
        cell("code", tunnel),
        cell("code", CLEANUP),
    ]
    # Reuse generic writer but retain unique, stable notebook cell IDs.
    write("05_gpu_qwen3_inference_cloudflare.ipynb", cells)
    output = ROOT / "notebooks/05_gpu_qwen3_inference_cloudflare.ipynb"
    book = json.loads(output.read_text(encoding="utf-8"))
    for index, item in enumerate(book["cells"]):
        item["id"] = f"gpu-qwen3-{index:03d}"
    output.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
