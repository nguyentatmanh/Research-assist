"""Build notebook 07: Qwen3-Reranker-4B QLoRA using the official SWIFT BCE recipe."""

import ast
import json
from pathlib import Path

from build_qwen_training_notebook import SETUP, without_repo_imports

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Fine-tune Qwen3-Reranker-4B — Kaggle T4, QLoRA, phiên 12 giờ

**Đây là notebook fine-tune reranker, file 07.** Chọn GPU T4, Internet ON, dùng
Save Version → Save & Run All trong một phiên mới. Model mặc định duy nhất là
`Qwen/Qwen3-Reranker-4B`. Một GPU được dùng; T4 thứ hai không tự cộng VRAM.

Theo hướng dẫn chính thức Qwen: `model_type=qwen3_reranker`,
`task_type=generative_reranker`, `loss_type=generative_reranker`. Loss là
**binary cross-entropy của logit yes − logit no**, label 1/0; không dùng InfoNCE
của embedding hoặc loss language modeling. Một group có 1 positive + 1 negative
và được SWIFT mở thành **2 query/document pairs**. Accumulation là 8 groups.

Điều chỉnh phần cứng so với ví dụ full train nhiều GPU của Qwen: **QLoRA NF4
4-bit**, double quantization, FP16, rank 8, SDPA, gradient checkpointing, batch
1 group và max_length 512. T4 không dùng BF16/FlashAttention 2. Chưa đo VRAM/thời
gian 4B trên Kaggle thật; nếu OOM, giảm MAX_LENGTH=256 rồi tạo run mới.

Tài liệu ví dụ dùng MTEB/scidocs-reranking; notebook này mặc định dùng QASPER để
khớp miền tài liệu nghiên cứu của project. Định dạng role, template và loss giữ đúng
SWIFT. Negative là silver, cần kiểm duyệt nếu muốn dữ liệu chất lượng cao hơn.

**Ngân sách từ đầu phiên, gồm setup/download/data:** dừng train ở giờ 10,
hard stop worker ở 10h30, dừng evaluation ở giờ 11, chừa giờ cuối lưu Output.
Lưu checkpoint mỗi 100 optimizer steps hoặc khoảng 15 phút. Không mở tunnel,
không chờ secret, không merge 8GB weights trong phiên train. Output giữ adapter
và base commit SHA, cùng checkpoint optimizer/RNG để resume.

Nguồn chính thức đã đối chiếu:
- https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
- https://huggingface.co/Qwen/Qwen3-Reranker-4B
- https://github.com/modelscope/ms-swift/tree/v3.12.3
- https://huggingface.co/docs/peft/developer_guides/quantization
"""

CONFIG = """import os, sys, json, time, subprocess, venv, importlib.metadata
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-Reranker-4B"
RUN_NAME = "qwen3-reranker-4b-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
MAX_LENGTH = 512  # total prompt + instruction + query + document + assistant prefix
LEARNING_RATE = 6e-6  # starting point from Qwen's example; tune only on validation
MAX_STEPS = 500  # total step cap, may stop earlier at the 10h deadline
PAIR_LIMIT = 6000  # number of groups; each group expands to two pairs
PREPARED_DATA = ""  # optional prepared corpus directory under /kaggle/input
RESUME_CHECKPOINT = ""  # previous Output / RUN / resume; recipe in its parent
SESSION_ALREADY_ELAPSED_HOURS = 0.0  # set actual age if the GPU session is older
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

DATA_NOTE = """## Dữ liệu, prompt và kiểm tra chất lượng

QASPER giữ split theo paper; positive train không trùng held-out text. Mỗi group
có query role user, tài liệu positive/negative role assistant; system chứa custom
instruction, được SWIFT đưa vào `<Instruct>`. Negative BM25 trong cùng paper,
loại mọi gold đã biết và held-out duplicate. Gold có thể chưa đầy đủ, nên negative
vẫn là silver. Corpus riêng theo định dạng prepared-data của project.

Giới hạn 512 token tính **toàn bộ prompt ghép**, không phải 512 token riêng document.
Loại group quá dài, không cắt mất gold. Trước train, kiểm tra token IDs của SWIFT
khớp hoàn toàn helper inference cho cả positive và negative, labels đúng `[1, 0]`.

Validation loss chọn checkpoint; sau train so base và adapter trên cùng panel
16 query / tối đa 20 BM25 candidates mỗi query. Không chèn gold vào candidates,
báo candidate recall, nDCG@10/MRR@10/Recall@10. Đây là kiểm tra nhỏ trên validation,
không dùng test chọn model và không suy ra chất lượng tiếng Việt từ QASPER tiếng Anh.
"""

PREPARE = """config = {
    "recipe": RECIPE, "run": str(RUN), "base_model": BASE_MODEL,
    "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS, "pair_limit": PAIR_LIMIT, "instruction": INSTRUCTION,
    "prepared_data": PREPARED_DATA, "resume": RESUME_CHECKPOINT,
    "output_dir": str(RUN / "checkpoints"),
    "train_deadline": TRAIN_DEADLINE, "eval_deadline": EVAL_DEADLINE,
}
assert BASE_MODEL == "Qwen/Qwen3-Reranker-4B"
assert 128 <= MAX_LENGTH <= 512 and MAX_STEPS > 0 and PAIR_LIMIT >= 16
assert SESSION_ALREADY_ELAPSED_HOURS >= 0 and INSTRUCTION.strip()
atomic_json(RUN / "config.json", config)
status = run_bounded([str(PYTHON), str(WORKER), "prepare", str(RUN / "config.json")],
    RUN / "prepare.log", min(TRAIN_DEADLINE - 1800, time.time() + 3600))
if status["returncode"] != 0:
    print((RUN / "prepare.log").read_text(encoding="utf-8")[-8000:])
    raise RuntimeError("Preparation failed/timed out. Inspect prepare.log before training.")
config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
stats = json.loads((RUN / "data_statistics.json").read_text(encoding="utf-8"))
print({k:v for k,v in stats.items() if k != "provenance"})
print("Prompt check:", json.loads((RUN / "prompt_check.json").read_text(encoding="utf-8")))
print("Pinned base revision:", config["base_revision"])
print("Train remaining hours:", round((TRAIN_DEADLINE - time.time()) / 3600, 2))
"""

TRAIN = """print("Training reranker; progress and logs are written continuously under", RUN)
train_status = run_bounded([str(PYTHON), str(WORKER), "train", str(RUN / "config.json")],
    RUN / "train.log", TRAIN_HARD_DEADLINE)
atomic_json(RUN / "process_status.json", train_status)
if train_status["returncode"] != 0:
    print((RUN / "train.log").read_text(encoding="utf-8")[-8000:])
    print("Preserving only complete, verified checkpoints if available.")
MODEL_OUT = export_reranker_adapter(config, train_status)
(MODEL_OUT / "qwen_reranker_scorer.py").write_text(QWEN_RERANKER_SCORER_SOURCE, encoding="utf-8")
shutil.make_archive(str(RUN / "qwen3-reranker-adapter"), "zip", MODEL_OUT)
print("Trained adapter:", MODEL_OUT)
print(json.loads((MODEL_OUT / "research_model.json").read_text(encoding="utf-8")))
"""

EVAL = """if time.time() < EVAL_DEADLINE - 300:
    evaluation_status = run_bounded(
        [str(PYTHON), str(WORKER), "evaluate", str(RUN / "config.json")],
        RUN / "evaluate.log", EVAL_DEADLINE)
    atomic_json(RUN / "evaluation_status.json", evaluation_status)
    result_path = RUN / "reranking_validation.json"
    if result_path.is_file():
        print(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        print("Evaluation incomplete/skipped; no partial metric is reported. See evaluate.log.")
else:
    atomic_json(RUN / "evaluation_status.json", {"skipped": "export reserve"})
    print("Evaluation skipped to preserve output time.")
"""

EXPORT = """from IPython.display import display, FileLink
for file in [RUN / "qwen3-reranker-adapter.zip", RUN / "config.json"]:
    display(FileLink(str(file)))
atomic_json(RUN / "session_report.json", {
    "session_limit_hours": 12, "train_stop_hours": 10, "train_hard_stop_hours": 10.5,
    "eval_stop_hours": 11, "elapsed_hours": (time.time() - SESSION_STARTED) / 3600,
    "optimizer_steps": read_reranker_manifest(MODEL_OUT)["optimizer_steps"],
    "resume_checkpoint": str(RUN / "resume"),
})
print("Download the adapter ZIP; retain the whole run folder for resume.")
print("Next session: add previous Output as Input, set RESUME_CHECKPOINT to its resume folder.")
print("Keep recipe/data unchanged; increase MAX_STEPS above the old global_step.")
print("Inference helper is included in the ZIP. Scores measure relevance, not factuality.")
"""

INFERENCE_NOTE = """## Dùng adapter sau khi train

Giải nén ZIP vào GPU host và dùng helper đi kèm, trong môi trường có các thư viện
đã pin ở trên. Base model đúng commit SHA sẽ được tải tự động; adapter nạp lỗi sẽ
báo lỗi, không âm thầm chuyển sang base model.

```python
import sys
sys.path.insert(0, "/path/to/extracted/qwen3-reranker-adapter")
from qwen_reranker_scorer import load_reranker_adapter
reranker = load_reranker_adapter("/path/to/extracted/qwen3-reranker-adapter", device="cuda:0")
scores = reranker.score([("research question", "candidate passage")], batch_size=1)
```

Đây là artifact **reranker**, có manifest/task riêng. Endpoint embedding trong
notebook 05 nhận artifact embedding; không dán adapter reranker vào đó.
Notebook này chưa triển khai API reranker/Cloudflare. Toàn bộ phiên Save & Run All
kết thúc sau khi lưu kết quả train và evaluation.
"""


def selected_nodes(source, names):
    nodes = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", "")
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            nodes.append(ast.get_source_segment(source, node))
    return "\n\n".join(nodes)


def build():
    data = (ROOT / "src/research_assist/experiments/data.py").read_text("utf-8")
    helper = (ROOT / "notebooks/kaggle_training_helpers.py").read_text("utf-8")
    common = (ROOT / "notebooks/kaggle_qwen_training.py").read_text("utf-8")
    common = selected_nodes(
        common,
        {
            "QWEN_REQUIREMENTS",
            "atomic_json",
            "session_start",
            "install_qwen_environment",
            "swift_arguments",
            "deadline_callback",
            "run_bounded",
            "complete_checkpoints",
        },
    )
    helper = selected_nodes(helper, {"download_qasper", "write_json", "NotebookBM25"})
    scorer = (ROOT / "notebooks/qwen_reranker_scorer.py").read_text("utf-8")
    training = without_repo_imports(
        (ROOT / "notebooks/kaggle_qwen_reranker_training.py").read_text("utf-8")
    )
    imports = (
        "import hashlib, json, os, shutil, subprocess, time, math, re, random\n"
        "from collections import Counter, defaultdict\nfrom pathlib import Path\n"
    )
    library = imports + data + "\n" + helper + "\n" + common + "\n" + scorer + "\n" + training
    main = """
if __name__ == "__main__":
    import sys
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    action = {"prepare": prepare_reranker_worker, "train": train_reranker_worker,
              "evaluate": evaluate_reranker_worker}
    action[sys.argv[1]](config)
"""
    setup = SETUP.replace("qwen_worker.py", "qwen_reranker_worker.py")
    setup = setup.replace("qwen_adapter_encoder.py", "qwen_reranker_scorer.py")
    setup = setup.replace("QWEN_ADAPTER_SOURCE", "QWEN_RERANKER_SCORER_SOURCE")
    sources = [
        ("markdown", INTRO),
        ("code", CONFIG),
        (
            "code",
            "QWEN_LIBRARY_SOURCE = "
            + repr(library)
            + "\nQWEN_WORKER_SOURCE = QWEN_LIBRARY_SOURCE + "
            + repr(main)
            + "\nQWEN_RERANKER_SCORER_SOURCE = "
            + repr(scorer)
            + "\nexec(QWEN_LIBRARY_SOURCE, globals())",
        ),
        ("code", setup),
        ("markdown", DATA_NOTE),
        ("code", PREPARE),
        ("code", TRAIN),
        ("code", EVAL),
        ("code", EXPORT),
        ("markdown", INFERENCE_NOTE),
    ]
    cells = []
    for index, (kind, source) in enumerate(sources):
        cell = {
            "cell_type": kind,
            "id": f"qwen4b-reranker-{index:03d}",
            "metadata": {"jupyter": {"source_hidden": True}} if index == 2 else {},
            "source": source.strip().splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
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
    path = ROOT / "notebooks/07_kaggle_finetune_qwen3_reranker_4b.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
