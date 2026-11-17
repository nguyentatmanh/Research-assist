"""Build the standalone Kaggle embedding worker notebook."""

import textwrap

from build_gpu_notebooks import ROOT, cell, write

INTRO = """# Research Assist · GPU embedding trên Kaggle

Chạy notebook này nếu bạn muốn **Research Assist trên máy mình gửi yêu cầu embedding
tới GPU Kaggle**. Bật **Internet**, chọn **GPU T4** và chạy trong phiên tương tác.
Thêm Kaggle Secret `RESEARCH_GPU_API_KEY` (ít nhất 32 ký tự) và cấp quyền cho notebook.
Nhập lại cùng khóa trong mục **GPU embedding** của Research Assist.

Mặc định dùng **Qwen3-Embedding-0.6B**, có thể chạy ngay. Để dùng LoRA từ notebook
**01**, gắn output làm Kaggle Input rồi trỏ `CHECKPOINT_ZIP` tới `qwen3-adapter.zip`.
Để dùng E5 đã huấn luyện, chọn `MODEL_FAMILY = "e5"` và cung cấp checkpoint E5.
Adapter **reranker** từ notebook 02 dành cho một tác vụ khác.

Bật `START_GPU_SERVER` và chạy lại từ cell cấu hình để cài thư viện, tải model
và kiểm tra API. Sau đó bật
`START_CLOUDFLARE` nếu bạn có quyền dùng đường hầm trên môi trường chạy.
URL tạm chỉ tồn tại trong phiên notebook. Khi xong, bật `STOP_ENDPOINT`
trong cell cuối để dừng server và đường hầm.
"""

CONFIG = """
MODEL_FAMILY = "qwen3"  # "qwen3" hoặc "e5"
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Đặt khớp model nếu dùng thư mục Qwen3 đầy đủ.
BASE_REVISION = ""  # Tùy chọn: SHA Hugging Face cho base model.
CHECKPOINT_DIR = ""  # Thư mục Qwen3 đầy đủ, SWIFT adapter, hoặc checkpoint E5.
CHECKPOINT_ZIP = ""  # VD: /kaggle/input/my-training/qwen3-adapter.zip.
E5_TOKENIZER_MODEL = "intfloat/multilingual-e5-base"  # Phải khớp checkpoint E5.
MAX_SEQ_LENGTH = 512
PORT = 8008
START_GPU_SERVER = False
START_CLOUDFLARE = False
API_KEY_SECRET_NAME = "RESEARCH_GPU_API_KEY"
"""

SETUP = """
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
WORKING_DIR = Path("/kaggle/temp/research-assist-embedding")
if START_GPU_SERVER:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "sentence-transformers==5.1.2", "transformers==4.57.1",
        "fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.27,<1",
        "huggingface-hub==0.36.0", "peft==0.17.1",
        "bitsandbytes==0.48.2", "accelerate==1.11.0",
    ])
"""

LIFECYCLE = """
def stop_embedding_worker():
    import gc

    tunnel = globals().pop("GPU_TUNNEL", None)
    try:
        if tunnel is not None:
            stop_process(tunnel)
    finally:
        server = globals().get("GPU_SERVER")
        if server is not None:
            server.should_exit = True
            GPU_THREAD.join(timeout=60)
            if GPU_THREAD.is_alive():
                raise RuntimeError("GPU đang hoàn tất yêu cầu. Thử lại cell dừng sau.")
            globals().pop("GPU_THREAD", None)
            globals().pop("GPU_SERVER", None)
            del server
            gc.collect()
            import torch

            torch.cuda.empty_cache()
        for name in ("GPU_API_KEY", "GPU_INFO", "GPU_PUBLIC_URL"):
            globals().pop(name, None)
"""

START = """
def start_embedding_worker():
    import tempfile

    import httpx
    import torch
    import uvicorn
    from huggingface_hub import snapshot_download
    from kaggle_secrets import UserSecretsClient

    global GPU_SERVER, GPU_THREAD, GPU_API_KEY, GPU_INFO

    if "GPU_SERVER" in globals():
        raise RuntimeError("Hãy dừng worker hiện tại trước khi chạy lại cell này.")
    if MODEL_FAMILY not in {"qwen3", "e5"} or not 64 <= MAX_SEQ_LENGTH <= 8192:
        raise ValueError("Kiểm tra MODEL_FAMILY và MAX_SEQ_LENGTH.")
    if CHECKPOINT_DIR and CHECKPOINT_ZIP:
        raise ValueError("Chỉ đặt một nguồn checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError("Bật GPU trong cài đặt Kaggle trước.")
    try:
        key = UserSecretsClient().get_secret(API_KEY_SECRET_NAME)
    except Exception:
        raise RuntimeError(f"Thêm và cấp quyền cho Kaggle Secret {API_KEY_SECRET_NAME}.") from None
    if not isinstance(key, str) or len(key) < 32 or not all("!" <= c <= "~" for c in key):
        raise ValueError("Dùng khóa ASCII từ 32 ký tự trở lên, ví dụ chuỗi hex ngẫu nhiên.")

    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORKING_DIR) as extracted:
        if CHECKPOINT_ZIP:
            unpack = extract_adapter_archive if MODEL_FAMILY == "qwen3" else extract_checkpoint
            selected = unpack(CHECKPOINT_ZIP, extracted)
        elif CHECKPOINT_DIR:
            selected = Path(CHECKPOINT_DIR)
        elif MODEL_FAMILY == "qwen3":
            if BASE_MODEL not in {"Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"}:
                raise ValueError("Chọn base Qwen3 Embedding 0.6B hoặc 4B.")
            selected = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION or None,
                                              ignore_patterns=["*.bin", "*.onnx", "onnx/*"]))
        else:
            raise ValueError("Chọn thư mục hoặc ZIP checkpoint E5 đã huấn luyện.")

        seq_length = MAX_SEQ_LENGTH
        tokenizer_model = E5_TOKENIZER_MODEL if MODEL_FAMILY == "e5" else BASE_MODEL
        if MODEL_FAMILY == "qwen3":
            manifest = selected / "research_model.json"
            if (manifest.is_file()
                    and json.loads(manifest.read_text(encoding="utf-8")).get("format")
                    == "swift_qwen3_adapter_v1"):
                adapter_info = read_adapter_manifest(selected)
                tokenizer_model = adapter_info["base_model"]
                seq_length = adapter_info["max_seq_length"]
            encoder, model_id = load_qwen_checkpoint(selected, "cuda", seq_length)
        else:
            encoder, model_id = load_e5_checkpoint(selected, "cuda", seq_length)
        app = create_embedding_app(
            encoder, key, model_id, seq_length, family=MODEL_FAMILY,
            tokenizer_model=tokenizer_model,
        )

    GPU_SERVER = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False,
    ))
    GPU_THREAD = threading.Thread(target=GPU_SERVER.run, daemon=True)
    try:
        GPU_THREAD.start()
        deadline = time.monotonic() + 60
        while not GPU_SERVER.started and GPU_THREAD.is_alive() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not GPU_SERVER.started:
            raise RuntimeError("Worker chưa khởi động. Kiểm tra cổng và log notebook.")
        headers = {"Authorization": "Bearer " + key}
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", trust_env=False) as client:
            client.get("/health", timeout=5).raise_for_status()
            if client.get("/v1/info", timeout=5).status_code != 401:
                raise RuntimeError("Worker chưa yêu cầu xác thực.")
            info = client.get("/v1/info", headers=headers, timeout=5).raise_for_status().json()
            for kind in ("query", "passage"):
                result = client.post("/v1/embeddings", headers=headers, timeout=120, json={
                    "model": model_id, "input": ["Research evidence"], "input_type": kind,
                }).raise_for_status().json()
                if len(result["data"][0]["embedding"]) != info["dimension"]:
                    raise RuntimeError("Kích thước vector không khớp thông tin checkpoint.")
        GPU_API_KEY, GPU_INFO = key, info
        print("GPU:", torch.cuda.get_device_name(0))
        print("Model ID:", info["model"])
        print("Số chiều:", info["dimension"])
        print("Đã kiểm tra API worker bằng yêu cầu query và passage.")
    except BaseException:
        if GPU_THREAD.ident is not None:
            stop_embedding_worker()
        else:
            del GPU_SERVER, GPU_THREAD
        raise


if START_GPU_SERVER:
    start_embedding_worker()
else:
    print("Đặt START_GPU_SERVER = True trong cell cấu hình để chạy worker.")
"""

TUNNEL = """
if START_CLOUDFLARE:
    import httpx

    if not globals().get("GPU_INFO") or not GPU_SERVER.started or not GPU_THREAD.is_alive():
        raise RuntimeError("Khởi động worker và kiểm tra API nội bộ trước.")
    if "GPU_TUNNEL" in globals() and GPU_TUNNEL.poll() is None:
        raise RuntimeError("Dừng tunnel hiện tại trước khi chạy lại cell này.")
    GPU_TUNNEL, url = start_quick_tunnel(PORT, WORKING_DIR)
    try:
        deadline = time.monotonic() + 45
        with httpx.Client(base_url=url, timeout=8) as client:
            while True:
                try:
                    health = client.get("/health")
                    if health.status_code == 200 and health.json() == {"status": "ok"}:
                        break
                except (httpx.RequestError, ValueError):
                    pass
                if GPU_TUNNEL.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Tunnel chưa kết nối được với worker Kaggle.")
                time.sleep(1)
            if client.get("/v1/info").status_code != 401:
                raise RuntimeError("Tunnel không trỏ tới worker có xác thực.")
            remote_info = client.get("/v1/info", headers={
                "Authorization": "Bearer " + GPU_API_KEY,
            }).raise_for_status().json()
            if remote_info != GPU_INFO:
                raise RuntimeError("Tunnel trả lại thông tin checkpoint không khớp.")
    except BaseException:
        stop_process(GPU_TUNNEL)
        del GPU_TUNNEL
        raise
    GPU_PUBLIC_URL = url
    print("URL GPU embedding:", GPU_PUBLIC_URL)
    print("Họ model:", GPU_INFO["family"], "· Tokenizer:", GPU_INFO["tokenizer_model"])
    print("Trong Research Assist, mở GPU embedding, chọn model/tokenizer hiển thị ở trên,")
    print("nhập URL và khóa trong Secret, kiểm tra rồi lưu. Tạo lại chỉ mục khi đổi checkpoint.")
else:
    print("Đặt START_CLOUDFLARE = True nếu cần kết nối từ máy khác.")
"""

CLEANUP = """
STOP_ENDPOINT = False
if STOP_ENDPOINT:
    stop_embedding_worker()
    print("Đã dừng worker GPU và tunnel.")
"""


def build():
    helpers = ROOT / "notebooks"
    server = (helpers / "gpu_embedding_server.py").read_text("utf-8")
    write(
        "03_gpu_server.ipynb",
        [
            cell("markdown", INTRO),
            cell("code", CONFIG),
            cell("code", SETUP),
            cell("code", (helpers / "kaggle_tunnel.py").read_text("utf-8")),
            cell("code", (helpers / "qwen_adapter_encoder.py").read_text("utf-8")),
            cell("code", "if START_GPU_SERVER:\n" + textwrap.indent(server, "    ")),
            cell("code", LIFECYCLE),
            cell("code", START),
            cell("code", TUNNEL),
            cell("code", CLEANUP),
        ],
        id_prefix="gpu-embedding",
    )


if __name__ == "__main__":
    build()
