"""Generate the single standalone Qwen3-4B fine-tuning entry point for Kaggle."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def without_repo_imports(source):
    lines = source.splitlines(keepends=True)
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith(("research_assist.", "notebooks."))
    ]
    for node in sorted(imports, key=lambda n: n.lineno, reverse=True):
        del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines)


INTRO = """# Fine-tune Qwen3-Embedding-4B trên Kaggle — QLoRA, tối đa 12 giờ

**Đây là notebook fine-tune Qwen bạn cần chạy.** GPU T4, Internet ON, Save Version
→ Save & Run All. Dùng đúng model `Qwen/Qwen3-Embedding-4B`, không dùng Qwen chat.
Một GPU được sử dụng; hai T4 không tạo thành một bộ nhớ chung.

Theo hướng dẫn Qwen: SWIFT, `task_type=embedding`, `model_type=qwen3_emb`, InfoNCE,
dataset `messages` / `positive_messages` / `negative_messages`. Chuyển từ ví dụ
full train nhiều GPU sang **QLoRA NF4 4-bit**, rank 8, batch 1, một explicit negative
để phù hợp T4; đây là cấu hình khởi đầu cần đo, chưa được cam kết GPU nào cũng đủ RAM.
Không bật BF16 hoặc FlashAttention 2 trên T4.

**Ngân sách tính cả cài đặt và tải dữ liệu/model:** dừng train ở giờ 10; hard stop
worker train ở 10h30; dừng đánh giá ở giờ 11; giữ giờ cuối cho đóng gói và Kaggle
lưu Output. Callback lưu mỗi 15 phút hoặc 100 optimizer steps; checkpoint có cả
optimizer/RNG để tiếp tục trong phiên sau. Giới hạn này giảm rủi ro mất kết quả,
không thể bảo đảm chống việc nền tảng tắt đột ngột hoặc tải mạng quá chậm.

Output chính là **adapter LoRA + base revision đã pin**, không merge 8GB weights
trong phiên train. Notebook 05 đã hỗ trợ nạp adapter này để chạy GPU endpoint.
Notebook 06 không mở tunnel hay chờ nhập key; Save & Run All sẽ kết thúc.

Tài liệu dùng để triển khai:
- https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
- https://github.com/modelscope/ms-swift/tree/v3.12.3
- https://huggingface.co/docs/peft/developer_guides/quantization
- https://huggingface.co/Qwen/Qwen3-Embedding-4B
"""

CONFIG = """import json, os, sys, time, subprocess, venv, importlib.metadata
from pathlib import Path

# Change these values before Save & Run All.
BASE_MODEL = "Qwen/Qwen3-Embedding-4B"
RUN_NAME = "qwen3-4b-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
MAX_LENGTH = 512  # reduce to 256 on OOM; start a new run if changing recipe
LEARNING_RATE = 2e-5
MAX_STEPS = 500  # upper cap, may stop earlier at the 10h deadline
PAIR_LIMIT = 6000
PREPARED_DATA = ""  # optional Research-assist prepared corpus under /kaggle/input
RESUME_CHECKPOINT = ""  # e.g. /kaggle/input/previous-run/RUN/resume
SESSION_ALREADY_ELAPSED_HOURS = 0.0  # enter session age if GPU was started earlier
INSTRUCTION = "Given a research question, retrieve relevant passages that answer the question"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/kaggle/temp/qwen-hf"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/temp/qwen-datasets"
os.environ["MODELSCOPE_CACHE"] = "/kaggle/temp/qwen-ms"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["NPROC_PER_NODE"] = "1"
"""

SETUP = '''# A private environment avoids changing Kaggle's already-loaded native libraries.
# Keep Kaggle's CUDA-enabled PyTorch via system-site-packages and an exact constraint.
SESSION_STARTED = session_start(
    "/kaggle/temp/qwen-session-clock.json", SESSION_ALREADY_ELAPSED_HOURS)
TRAIN_DEADLINE = SESSION_STARTED + 10 * 3600
TRAIN_HARD_DEADLINE = SESSION_STARTED + 10.5 * 3600
EVAL_DEADLINE = SESSION_STARTED + 11 * 3600
RUN = Path("/kaggle/working") / RUN_NAME
assert not RUN.exists(), "Choose a new RUN_NAME; never overwrite a previous run."
RUN.mkdir(parents=True)
ENV = Path("/kaggle/temp/qwen-training-env")
venv.EnvBuilder(with_pip=False, system_site_packages=True).create(ENV)
PYTHON = ENV / "bin/python"
constraints = ENV / "torch-constraints.txt"
torch_version = importlib.metadata.version("torch")
constraints.write_text("torch==" + torch_version + "\\n", encoding="utf-8")
install_qwen_environment(PYTHON, QWEN_REQUIREMENTS, constraints, TRAIN_DEADLINE)
WORKER = Path("/kaggle/temp/qwen_worker.py")
WORKER.write_text(QWEN_WORKER_SOURCE, encoding="utf-8")
ADAPTER_MODULE = Path("/kaggle/temp/qwen_adapter_encoder.py")
ADAPTER_MODULE.write_text(QWEN_ADAPTER_SOURCE, encoding="utf-8")
preflight = r"""
import importlib.metadata as m, torch, numpy, pandas, datasets, bitsandbytes
from swift.llm import TrainArguments, sft_main
from swift.plugin import extra_callbacks
from transformers import Qwen3Model
from peft import PeftModel
from packaging.version import Version
assert torch.cuda.is_available(), "Enable Kaggle T4 GPU before Save Version"
assert Version(torch.__version__.split("+")[0]) >= Version("2.6"), (
    "Use a Kaggle image with PyTorch >=2.6 for safe optimizer-state resume")
assert torch.cuda.get_device_properties(0).total_memory >= 14 * 2**30
print("GPU:", torch.cuda.get_device_name(0))
print({p:m.version(p) for p in ["torch","ms-swift","transformers","peft","bitsandbytes"]})
"""
subprocess.run([str(PYTHON), "-c", preflight], check=True, timeout=180)
print("Session elapsed hours:", round((time.time() - SESSION_STARTED) / 3600, 2))
'''

DATA_NOTE = """## Dữ liệu

Mặc định tải QASPER chính thức, giữ split theo paper; không gọi LLM tạo dữ liệu.
QASPER chủ yếu tiếng Anh, nên kết quả này chưa chứng minh chất lượng tiếng Việt.
Corpus riêng dùng cùng định dạng prepared-data của notebook E5. Một negative
được chọn theo overlap từ trong cùng paper, loại mọi gold đã biết; các negative
này là **silver**, vẫn có thể chứa đoạn liên quan chưa được gán nhãn. Các mẫu quá
dài bị loại trước khi train để không cắt mất gold. Test split không dùng chọn model.

Batch 1 có một positive + một negative. Gradient accumulation 8 chỉ tăng số mẫu
giữa hai lần cập nhật optimizer; không tăng số negative của một phép InfoNCE.
Validation loss chọn checkpoint. Có thêm phép đo retrieval trên một panel cố định
16 query / tối đa khoảng 256 đoạn validation, cùng quantization trước/sau; đó là
kiểm tra nhỏ, không đại diện toàn bộ benchmark.
"""

PREPARE = """config = {
    "run": str(RUN), "base_model": BASE_MODEL, "base_revision": "",
    "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS, "pair_limit": PAIR_LIMIT, "instruction": INSTRUCTION,
    "prepared_data": PREPARED_DATA, "resume": RESUME_CHECKPOINT,
    "output_dir": str(RUN / "checkpoints"),
    "train_deadline": TRAIN_DEADLINE, "eval_deadline": EVAL_DEADLINE,
}
assert BASE_MODEL in {"Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B"}
assert 80 <= MAX_LENGTH <= 512 and MAX_STEPS > 0 and PAIR_LIMIT >= 16
atomic_json(RUN / "config.json", config)
status = run_bounded([str(PYTHON), str(WORKER), "prepare", str(RUN / "config.json")],
    RUN / "prepare.log", min(TRAIN_DEADLINE - 1800, time.time() + 3600))
if status["returncode"] != 0:
    print((RUN / "prepare.log").read_text(encoding="utf-8")[-8000:])
    raise RuntimeError("Preparation failed or timed out. Inspect prepare.log in Output.")
config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
print(json.loads((RUN / "triplet_statistics.json").read_text(encoding="utf-8")))
print("Pinned base revision:", config["base_revision"])
print("Train remaining hours:", round((TRAIN_DEADLINE - time.time()) / 3600, 2))
"""

TRAIN = """# Fresh child process owns GPU memory and can be drained before evaluation/export.
print("Training; logs and progress are written continuously under", RUN)
train_status = run_bounded([str(PYTHON), str(WORKER), "train", str(RUN / "config.json")],
    RUN / "train.log", TRAIN_HARD_DEADLINE)
atomic_json(RUN / "process_status.json", train_status)
if train_status["returncode"] != 0:
    print((RUN / "train.log").read_text(encoding="utf-8")[-8000:])
    print("Attempting to preserve the last complete checkpoint.")
MODEL_OUT = export_adapter(config, train_status)
# Secure the small deploy artifact before any optional evaluation.
shutil.make_archive(str(RUN / "qwen3-adapter"), "zip", MODEL_OUT)
print("Trained adapter saved:", MODEL_OUT)
print(json.loads((MODEL_OUT / "research_model.json").read_text(encoding="utf-8")))
"""

EVAL = """if time.time() < EVAL_DEADLINE - 300:
    evaluation_status = run_bounded(
        [str(PYTHON), str(WORKER), "evaluate", str(RUN / "config.json")],
        RUN / "evaluate.log", EVAL_DEADLINE)
    atomic_json(RUN / "evaluation_status.json", evaluation_status)
    result_path = RUN / "retrieval_validation.json"
    if result_path.is_file():
        print(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        print("Evaluation incomplete/skipped; no partial metric is reported. See evaluate.log.")
else:
    atomic_json(RUN / "evaluation_status.json", {"skipped": "export reserve"})
    print("Evaluation skipped to preserve output time.")
"""

EXPORT = """# Keep deploy adapter and resume checkpoint separate. No 8GB merge is required.
from IPython.display import display, FileLink
for file in [RUN / "qwen3-adapter.zip", RUN / "config.json"]:
    display(FileLink(str(file)))
atomic_json(RUN / "session_report.json", {
    "session_limit_hours": 12, "train_stop_hours": 10, "train_hard_stop_hours": 10.5,
    "eval_stop_hours": 11, "elapsed_hours": (time.time() - SESSION_STARTED) / 3600,
    "optimizer_steps": read_adapter_manifest(MODEL_OUT)["optimizer_steps"],
    "resume_checkpoint": str(RUN / "resume"),
})
print("Save Version Output contains the adapter ZIP, logs, data, metrics and resume checkpoint.")
print("Next session: add this Output as Input; set RESUME_CHECKPOINT to its resume directory.")
print("To serve: add adapter ZIP as Input, open notebook 05 and set ADAPTER_ZIP.")
print("Use Qwen3 family in the app; changing checkpoint requires rebuilding document vectors.")
"""


def build():
    data_source = (ROOT / "src/research_assist/experiments/data.py").read_text("utf-8")
    helper_source = (ROOT / "notebooks/kaggle_training_helpers.py").read_text("utf-8")
    # Embed only the official QASPER downloader and its small writer helper.
    nodes = [
        node
        for node in ast.parse(helper_source).body
        if isinstance(node, ast.FunctionDef) and node.name in {"download_qasper", "write_json"}
    ]
    download_source = "\n\n".join(ast.get_source_segment(helper_source, node) for node in nodes)
    training = without_repo_imports((ROOT / "notebooks/kaggle_qwen_training.py").read_text("utf-8"))
    adapter = (ROOT / "notebooks/qwen_adapter_encoder.py").read_text("utf-8")
    combined = data_source + "\n" + download_source + "\n" + training + "\n" + adapter
    main = """
if __name__ == "__main__":
    import sys
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    action = {"prepare": prepare_worker, "train": train_worker, "evaluate": evaluate_worker}
    action[sys.argv[1]](config)
"""
    sources = [
        ("markdown", INTRO),
        ("code", CONFIG),
        (
            "code",
            "QWEN_LIBRARY_SOURCE = "
            + repr(combined)
            + "\nQWEN_WORKER_SOURCE = QWEN_LIBRARY_SOURCE + "
            + repr(main)
            + "\nQWEN_ADAPTER_SOURCE = "
            + repr(adapter)
            + "\nexec(QWEN_LIBRARY_SOURCE, globals())",
        ),
        ("code", SETUP),
        ("markdown", DATA_NOTE),
        ("code", PREPARE),
        ("code", TRAIN),
        ("code", EVAL),
        ("code", EXPORT),
    ]
    cells = []
    for index, (kind, source) in enumerate(sources):
        item = {
            "cell_type": kind,
            "id": f"qwen4b-train-{index:03d}",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    notebook = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
    }
    path = ROOT / "notebooks/06_kaggle_finetune_qwen3_4b.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
