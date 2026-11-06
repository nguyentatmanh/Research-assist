"""Build standalone notebooks; their outputs need no repository upload on Kaggle."""

import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, *, hidden=False):
    value = {
        "cell_type": kind,
        "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def md(source):
    return cell("markdown", source)


def code(source, *, requires_ready=True, **kwargs):
    source = textwrap.dedent(source).strip()
    if requires_ready:
        # Return a visible restart notice from setup, and safely skip every later
        # cell until a fresh kernel has completed setup. Do not raise a fake error.
        source = 'if globals().get("_RESEARCHDESK_ENV_READY", False):\n' + textwrap.indent(
            source, "    "
        )
    return cell("code", source, **kwargs)


SETUP = r"""
import os, sys, subprocess, time
_RESEARCHDESK_ENV_READY = False
# This timer includes dependency installation and data/model preparation.
# If GPU/session started before this kernel (including a restart), enter its elapsed age.
SESSION_ALREADY_ELAPSED_HOURS = 0.0
clock_path = Path("/kaggle/temp/researchdesk-session-clock.json")
if "_NOTEBOOK_STARTED_MONOTONIC" not in globals():
    try:
        import psutil
        kernel_age = max(0.0, time.time() - psutil.Process(os.getpid()).create_time())
    except ImportError:
        kernel_age = 0.0
    _NOTEBOOK_STARTED_MONOTONIC = time.monotonic() - max(
        kernel_age, SESSION_ALREADY_ELAPSED_HOURS * 3600
    )
    if clock_path.is_file():
        saved_start = json.loads(clock_path.read_text())["started_epoch"]
        _NOTEBOOK_STARTED_MONOTONIC = min(
            _NOTEBOOK_STARTED_MONOTONIC, time.monotonic() - max(0, time.time() - saved_start)
        )
_NOTEBOOK_STARTED_MONOTONIC = min(
    _NOTEBOOK_STARTED_MONOTONIC, time.monotonic() - SESSION_ALREADY_ELAPSED_HOURS * 3600
)
clock_path.parent.mkdir(parents=True, exist_ok=True)
clock_path.write_text(json.dumps({
    "started_epoch": time.time() - (time.monotonic() - _NOTEBOOK_STARTED_MONOTONIC)
}))
# Run in a fresh Kaggle GPU session, before importing torch/transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HOME"] = "/kaggle/temp/researchdesk-hf"
# Keep Kaggle's CUDA PyTorch. Verify imports in a clean child process first.
environment_status = install_training_environment(
    "/kaggle/temp/researchdesk-environment.json"
)
_RESEARCHDESK_ENV_READY = environment_status["ready"]
"""

ENVIRONMENT = r"""
import gc, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, CrossEncoder
assert torch.cuda.is_available(), "Enable a GPU in Kaggle Settings before training."
print("Python:", sys.version)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("Packages:", json.dumps(package_versions(), indent=2))
print("One GPU is used. Two T4 cards do not form one shared VRAM pool.")
"""

DATA = r"""
DATA = Path(PREPARED_DATA) if PREPARED_DATA else Path("/kaggle/temp/researchdesk-qasper-prepared")
if not PREPARED_DATA:
    raw = Path("/kaggle/temp/researchdesk-qasper-raw")
    download_info = download_qasper(raw, QASPER_REVISION)
    dataset_manifest = prepare_qasper(raw, DATA)
    write_json(RUN / "experiment" / "download.json", download_info)
else:
    required = ["corpus.jsonl", "train_pairs.jsonl", "manifest.json"]
    required += [f"{s}_queries.jsonl" for s in ("train", "validation", "test")]
    assert all((DATA / name).is_file() for name in required), "Incomplete prepared dataset"
    dataset_manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    # Revalidate IDs, split provenance and exact duplicate evidence before any training.
    validated_pairs(DATA, Path(SYNTHETIC_PATH) if SYNTHETIC_PATH else None)
for split in ("train", "validation", "test"):
    corpus, questions = load_split(DATA, split)
    print(split, "paragraphs:", len(corpus), "eligible questions:", len(questions))
print("Dataset statistics:", dataset_manifest.get("statistics", {}))
write_json(RUN / "experiment" / "dataset_manifest.json", dataset_manifest)
write_json(RUN / "experiment" / "packages.json", package_versions())
"""

DATA_NOTE = """
### Dữ liệu và chia tập
Mặc định tải **QASPER chính thức, tiếng Anh**, pin commit SHA của bản Parquet;
không gọi LLM API. Giữ nguyên split theo paper, loại gold hình/bảng, gold không
khớp đầy đủ và positive train trùng text held-out. Validation chọn checkpoint;
test chỉ chạy một lần sau khi chốt cấu hình. Không kết luận chất lượng tiếng Việt từ QASPER.

Để dùng corpus riêng, đặt `PREPARED_DATA` tới thư mục Kaggle Input chứa
`corpus.jsonl`, `train_pairs.jsonl`, `{train,validation,test}_queries.jsonl`, `manifest.json`
theo định dạng Research Desk. Split tài liệu trước khi tạo câu hỏi. Có thể thêm
JSONL train tiếng Việt bằng `SYNTHETIC_PATH`: mỗi dòng có `question`, `positive_id`,
`paper_id`, `reviewed=true`; đoạn positive phải thuộc corpus train, được người đọc kiểm duyệt.
"""

TIME_CONFIG = r"""
# All limits are measured from the first setup cell (or the older kernel/session age).
SESSION_LIMIT_HOURS = 12
TRAIN_STOP_AFTER_HOURS = 10
EXPORT_RESERVE_HOURS = 1  # evaluation stops at hour 11; preserve the last hour for export
CHECKPOINT_EVERY_MINUTES = 15
TIME_BUDGET = NotebookTimeBudget(
    started=_NOTEBOOK_STARTED_MONOTONIC, session_hours=SESSION_LIMIT_HOURS,
    train_stop_hours=TRAIN_STOP_AFTER_HOURS, export_reserve_hours=EXPORT_RESERVE_HOURS,
)
write_json(RUN / "experiment" / "time_budget.json", TIME_BUDGET.snapshot())
print("Session time budget:", json.dumps(TIME_BUDGET.snapshot(), indent=2))
"""

E5_CONFIG = r"""
# Edit only these settings for a first experiment.
BASE_MODEL = "intfloat/multilingual-e5-base"  # same size as the current app baseline
BASE_REVISION = ""  # blank resolves and records a commit SHA
RUN_NAME = "e5-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
EPOCHS = 2
BATCH_SIZE = 32             # logical contrastive batch, not all encoded at once
MINI_BATCH_SIZE = 4         # CachedMNRL encodes small groups to reduce VRAM
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
SEED = 42
QASPER_REVISION = ""
PREPARED_DATA = ""         # optional prepared dataset directory under /kaggle/input
SYNTHETIC_PATH = ""        # optional HUMAN-REVIEWED training JSONL
RUN_FINAL_TEST = False     # set True only after freezing all hyperparameters
RUN = Path("/kaggle/working") / RUN_NAME
assert not RUN.exists(), "Choose a new RUN_NAME; do not overwrite an experiment."
(RUN / "experiment").mkdir(parents=True)
MODEL_OUT = RUN / "model"
"""

E5_TRAIN = r"""
revision = BASE_REVISION or HfApi().model_info(BASE_MODEL).sha
config = TrainConfig(
    data=str(DATA), output=str(MODEL_OUT), model=BASE_MODEL, revision=revision,
    epochs=EPOCHS, batch_size=BATCH_SIZE, mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LEARNING_RATE, max_seq_length=MAX_SEQ_LENGTH,
    device="cuda", seed=SEED, synthetic=SYNTHETIC_PATH or None,
)
write_json(RUN / "experiment" / "run_config.json", asdict(config))
manifest = train(config, time_budget=TIME_BUDGET, checkpoint_minutes=CHECKPOINT_EVERY_MINUTES)
print("Training status:", {key: manifest[key] for key in (
    "fine_tuned", "optimizer_steps", "selection", "stop_reason"
)})
write_json(RUN / "experiment" / "validation.json", {
    "before": manifest["validation_before"], "after": manifest["validation_after"],
})
print(json.dumps({key: manifest[key] for key in (
    "pairs_used", "pairs_excluded_truncation", "validation_before", "validation_after"
)}, indent=2))
# These are retrieval scores, not accuracy of generated answers. A fine-tune can regress.
gc.collect()
torch.cuda.empty_cache()
"""

E5_TEST = r"""
if RUN_FINAL_TEST and not TIME_BUDGET.training_expired():
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    corpus, queries = load_split(DATA, "test")
    evaluator = InformationRetrievalEvaluator(
        queries={q["id"]: "query: " + q["question"] for q in queries},
        corpus={c["id"]: "passage: " + c["text"] for c in corpus},
        relevant_docs={q["id"]: set(q["relevant_ids"]) for q in queries},
        name="test", batch_size=16, show_progress_bar=True,
    )
    test_report = {"split": "test", "corpus_sha256": file_hash(DATA / "corpus.jsonl"),
                   "queries_sha256": file_hash(DATA / "test_queries.jsonl"), "models": {}}
    for label, name in (("base", BASE_MODEL), ("fine_tuned", str(MODEL_OUT))):
        kwargs = {"revision": revision} if label == "base" else {}
        test_model = SentenceTransformer(name, device="cuda", trust_remote_code=False, **kwargs)
        test_model.max_seq_length = MAX_SEQ_LENGTH
        try:
            test_report["models"][label] = evaluate_with_budget(
                test_model, lambda: evaluator(test_model), TIME_BUDGET
            )
        except NotebookDeadlineReached:
            test_report["incomplete_due_to_time_budget"] = True
            break
        del test_model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(RUN / "experiment" / "test.json", test_report)
    print(json.dumps(test_report, indent=2))
else:
    print("Test skipped: disabled or insufficient time. Freeze settings before final test.")
"""

E5_RELOAD = r"""
# Offline CPU reload: this checks packaging/load compatibility, not research quality.
reload_model = SentenceTransformer(str(MODEL_OUT), device="cpu", local_files_only=True)
vectors = reload_model.encode(
    ["query: How is evidence retrieved?", "passage: Evidence is retrieved from relevant passages."],
    normalize_embeddings=True, convert_to_numpy=True,
)
assert vectors.shape[0] == 2 and np.isfinite(vectors).all()
assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
write_json(RUN / "experiment" / "reload_check.json", {
    "passed": True, "device": "cpu", "embedding_dimension": int(vectors.shape[1]),
})
del reload_model
gc.collect()
"""

RERANK_CONFIG = r"""
BASE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BASE_REVISION = ""
RUN_NAME = "reranker-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
EPOCHS = 2
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
NEGATIVES_PER_QUERY = 3
CANDIDATES = 40
SEED = 42
QASPER_REVISION = ""
PREPARED_DATA = ""
SYNTHETIC_PATH = ""       # optional human-reviewed VI training positives
REVIEWED_NEGATIVES = ""   # optional JSONL: query_id, negative_id, label=0, reviewed=true
RETRIEVER_PATH = ""       # optional E5 checkpoint directory added as Kaggle Input
RETRIEVER_ZIP = ""        # or checkpoint ZIP from notebook 01 (not experiment ZIP)
RUN_FINAL_TEST = False
RUN = Path("/kaggle/working") / RUN_NAME
assert not RUN.exists(), "Choose a new RUN_NAME; do not overwrite an experiment."
(RUN / "experiment").mkdir(parents=True)
MODEL_OUT = RUN / "model"
"""

RERANK_PREPARE = r"""
if RETRIEVER_ZIP:
    assert not RETRIEVER_PATH, "Choose RETRIEVER_ZIP or RETRIEVER_PATH, not both."
    RETRIEVER_PATH = unzip_checkpoint(RETRIEVER_ZIP, Path("/kaggle/temp") / (RUN_NAME + "-e5"))
if RETRIEVER_PATH:
    assert (Path(RETRIEVER_PATH) / "modules.json").is_file(), "Need a full E5 checkpoint"
    retrieval_metadata = {"mode": "BM25+E5 RRF", "model_path": RETRIEVER_PATH,
                          "config_sha256": file_hash(Path(RETRIEVER_PATH) / "config.json")}
    source_manifest = Path(RETRIEVER_PATH) / "research_model.json"
    if source_manifest.is_file():
        retrieval_metadata["training_manifest"] = json.loads(source_manifest.read_text())
else:
    retrieval_metadata = {"mode": "BM25", "model_path": None}
rows, review_rows = make_reranker_pairs(
    DATA, negatives_per_query=NEGATIVES_PER_QUERY, seed=SEED,
    reviewed_path=REVIEWED_NEGATIVES, synthetic=SYNTHETIC_PATH or None,
)
write_jsonl(RUN / "experiment" / "training_pairs.jsonl", rows)
write_jsonl(RUN / "experiment" / "negative_review.jsonl", review_rows)
print("Training rows:", len(rows), "Sources:", dict(Counter(r["source"] for r in rows)))
print("Negatives to review:", RUN / "experiment" / "negative_review.jsonl")
try:
    validation_samples = build_candidates(
        DATA, "validation", count=CANDIDATES, retriever_path=RETRIEVER_PATH,
        batch_size=BATCH_SIZE, device="cuda", time_budget=TIME_BUDGET,
    )
except NotebookDeadlineReached:
    validation_samples = []
    print("Validation candidates incomplete due to time budget; proceeding to safe export.")
write_json(RUN / "experiment" / "retrieval_configuration.json", retrieval_metadata)
write_json(RUN / "experiment" / "validation_candidates.json", validation_samples)
retrieval_baseline = evaluate_candidates(None, validation_samples) if validation_samples else {
    "metrics": None, "skipped_due_to_time_budget": True,
}
write_json(RUN / "experiment" / "validation_retrieval_only.json", retrieval_baseline)
print("Retrieval-only validation:", retrieval_baseline["metrics"])
"""

RERANK_TRAIN = r"""
revision = BASE_REVISION or HfApi().model_info(BASE_MODEL).sha
manifest = train_reranker(
    BASE_MODEL, MODEL_OUT, rows, validation_samples, revision=revision,
    epochs=EPOCHS, batch_size=BATCH_SIZE, accumulation=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE, max_length=MAX_SEQ_LENGTH, seed=SEED,
    time_budget=TIME_BUDGET, checkpoint_minutes=CHECKPOINT_EVERY_MINUTES,
)
print("Training status:", {key: manifest[key] for key in (
    "fine_tuned", "optimizer_steps", "selection", "stop_reason"
)})
manifest.update({
    "corpus_sha256": file_hash(DATA / "corpus.jsonl"),
    "training_pairs_sha256": file_hash(RUN / "experiment" / "training_pairs.jsonl"),
    "validation_candidates_sha256": file_hash(RUN / "experiment" / "validation_candidates.json"),
    "retrieval_configuration": retrieval_metadata,
    "negative_policy": "human_reviewed" if REVIEWED_NEGATIVES else "cross_paper_weak",
    "synthetic_sha256": file_hash(Path(SYNTHETIC_PATH)) if SYNTHETIC_PATH else None,
})
write_json(MODEL_OUT / "research_model.json", manifest)
for name in ("validation_before.json", "validation_after.json", "training_history.json"):
    shutil.copy2(MODEL_OUT / name, RUN / "experiment" / name)
print("Base reranker:", manifest["validation_before"])
print("Fine-tuned reranker:", manifest["validation_after"])
print("Best epoch:", manifest["best_epoch"])
gc.collect()
torch.cuda.empty_cache()
"""

RERANK_TEST = r"""
if RUN_FINAL_TEST and not TIME_BUDGET.training_expired():
    try:
        samples = build_candidates(DATA, "test", count=CANDIDATES,
                                   retriever_path=RETRIEVER_PATH, batch_size=BATCH_SIZE,
                                   time_budget=TIME_BUDGET)
    except NotebookDeadlineReached:
        samples = []
    test_report = {"split": "test", "gold_injection": False,
                   "retrieval_configuration": retrieval_metadata,
                   "queries_sha256": file_hash(DATA / "test_queries.jsonl"),
                   "retrieval_only": evaluate_candidates(None, samples) if samples else None,
                   "incomplete_due_to_time_budget": not bool(samples), "models": {}}
    for label, name in ((("base", BASE_MODEL), ("fine_tuned", str(MODEL_OUT))) if samples else []):
        kwargs = {"revision": revision} if label == "base" else {}
        test_model = CrossEncoder(name, device="cuda", max_length=MAX_SEQ_LENGTH,
                                  trust_remote_code=False, **kwargs)
        try:
            test_report["models"][label] = evaluate_with_budget(
                test_model, lambda: evaluate_candidates(test_model, samples, BATCH_SIZE),
                TIME_BUDGET,
            )
        except NotebookDeadlineReached:
            test_report["incomplete_due_to_time_budget"] = True
            break
        del test_model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(RUN / "experiment" / "test.json", test_report)
    print({name: report["metrics"] for name, report in test_report["models"].items()})
else:
    print("Test skipped: disabled or insufficient time. Do not tune on the test split.")
"""

RERANK_RELOAD = r"""
reload_model = CrossEncoder(str(MODEL_OUT), device="cpu", max_length=MAX_SEQ_LENGTH,
                            local_files_only=True)
scores = reload_model.predict([
    ("How is evidence retrieved?", "Evidence is retrieved from relevant passages."),
    ("How is evidence retrieved?", "This paragraph describes a cooking recipe."),
], show_progress_bar=False)
assert len(scores) == 2 and np.isfinite(scores).all()
write_json(RUN / "experiment" / "reload_check.json", {
    "passed": True, "device": "cpu", "score_count": len(scores),
})
print("Offline CPU reload passed. Example scores:", scores)
# Scores are ranking signals, not calibrated factuality probabilities.
del reload_model
gc.collect()
"""


def export_cell(role):
    return code(f"""
        checkpoint_zip = Path("/kaggle/working/research-{role}-checkpoint.zip")
        experiment_zip = Path("/kaggle/working/research-{role}-experiment.zip")
        paths = export_checkpoint(MODEL_OUT, checkpoint_zip, RUN / "experiment", experiment_zip)
        from IPython.display import FileLink, display
        for path in paths:
            print(path)
            display(FileLink(path))
        print("Download both ZIPs from Kaggle Output after saving the notebook version.")
    """)


def build():
    environment_source = (ROOT / "notebooks/kaggle_environment.py").read_text(encoding="utf-8")
    data_source = (ROOT / "src/research_assist/experiments/data.py").read_text(encoding="utf-8")
    helper_source = (ROOT / "notebooks/kaggle_training_helpers.py").read_text(encoding="utf-8")
    # The data helpers are already embedded in the preceding cell.
    lines = helper_source.splitlines(keepends=True)
    for node in reversed(ast.parse(helper_source).body):
        if isinstance(node, ast.ImportFrom) and node.module == "research_assist.experiments.data":
            del lines[node.lineno - 1 : node.end_lineno]
    helper_source = "".join(lines)
    train_source = (ROOT / "notebooks/kaggle_e5_training.py").read_text(encoding="utf-8")
    lines = train_source.splitlines(keepends=True)
    for node in reversed(ast.parse(train_source).body):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "research_assist.experiments.data",
            "notebooks.kaggle_training_helpers",
        }:
            del lines[node.lineno - 1 : node.end_lineno]
    train_source = "".join(lines)
    common = [
        md(
            "### 1. Cài thư viện\nBật **Internet** và **GPU** trong Kaggle Settings. "
            "Chạy notebook từ đầu trong session mới; không cần key hay source ZIP. "
            "Nếu cell báo **THIẾT LẬP ĐÃ THÀNH CÔNG / RESTART KERNEL**, "
            "restart kernel/Python một lần, "
            "rồi chạy lại từ cell đầu. Không cần tạo lại phiên GPU. "
            "Đây là bước bắt buộc để tránh dùng chung NumPy cũ trong RAM và bản vừa cài trên đĩa."
        ),
        code(environment_source + "\n" + SETUP, requires_ready=False),
        md(
            "**Ngân sách thời gian:** dừng tối ưu ở giờ 10, dừng đánh giá ở giờ 11, "
            "dành giờ cuối để lưu và đóng gói. Timer tính cả cài đặt/tải dữ liệu. "
            "Nếu GPU/session đã chạy trước kernel này, đặt SESSION_ALREADY_ELAPSED_HOURS "
            "trong cell đầu. Đây là dừng mềm giữa các batch, không bảo đảm trước sự cố "
            "hoặc khi Kaggle cưỡng chế ngắt session. Lưu Kaggle Version/Output để giữ artifact."
        ),
        md("### 2. Hàm hỗ trợ\nCell này chứa mã đầy đủ, đã nhúng sẵn để notebook chạy độc lập."),
        code(data_source, hidden=True),
        code(helper_source, hidden=True),
        code(ENVIRONMENT),
    ]
    e5 = [
        md(
            "# 01 · Fine-tune E5 retriever cho Research Desk\n"
            "Học câu hỏi → đoạn bằng chứng bằng **CachedMultipleNegativesRankingLoss**. "
            "Mặc định E5-base, một GPU, logical batch 32 / mini-batch 4. "
            "Đây là notebook để bạn thực thi, chưa kèm kết quả GPU đã chạy."
        ),
        *common,
        md(
            "### 3. Cấu hình\nNếu OOM, giảm `MINI_BATCH_SIZE` còn 2 rồi 1 và dùng RUN_NAME mới; "
            "có thể chọn E5-small nếu cần giảm tải. So sánh trước–sau phải cùng model nền. "
            "Logical batch khác gradient accumulation; CachedMNRL giữ tập negatives rộng hơn."
        ),
        code(E5_CONFIG + TIME_CONFIG),
        md(DATA_NOTE),
        code(DATA),
        md(
            "### 4. Fine-tune và validation\nKhông reset giới hạn 512 token. "
            "Loại positive bị truncation, dùng NO_DUPLICATES sampler; có thể còn false negatives "
            "giữa các đoạn tương đương. Chọn epoch theo metric validation, ghi hash và seed. "
            "Giữ toàn bộ thư mục run nếu muốn lưu cả checkpoint trung gian."
        ),
        code(train_source, hidden=True),
        code(E5_TRAIN),
        md(
            "### 5. Test cuối — mặc định tắt\nChỉ bật `RUN_FINAL_TEST` sau khi chốt cấu hình. "
            "So sánh E5 nền và fine-tuned trên cùng toàn bộ corpus test; "
            "không chọn model bằng test."
        ),
        code(E5_TEST),
        md(
            "### 6. Nạp lại offline trên CPU và xuất model\n"
            "Kiểm tra này xác nhận checkpoint nạp được; "
            "chất lượng phải đọc từ metrics và đánh giá câu trả lời cuối."
        ),
        code(E5_RELOAD),
        export_cell("e5"),
        md("""
        ### 7. Đưa vào hệ thống
        Tải `research-e5-checkpoint.zip` và `research-e5-experiment.zip`.
        Giải nén checkpoint vào `models/multilingual-e5-research`; `modules.json`,
        `config.json`, tokenizer, weights và `research_model.json` phải nằm đúng vị trí.

        ```dotenv
        RA_EMBEDDING_PROVIDER=sentence-transformers
        RA_EMBEDDING_MODEL=models/multilingual-e5-research
        RA_EMBEDDING_FAMILY=e5
        RA_EMBEDDING_MAX_SEQ_LENGTH=512
        RA_LOCAL_DEVICE=cpu
        ```

        Có thể dùng `cuda` nếu máy đủ VRAM. Restart backend và **Reindex toàn bộ embedding**.
        Không trộn vector của model cũ/mới. Notebook 02 có thể dùng ZIP checkpoint này làm Input.
        ZIP experiment là bằng chứng thực nghiệm, không phải model để nạp vào app.

        Một `.pt` riêng không chứa đủ tokenizer, pooling và cấu hình. Dùng nguyên checkpoint
        `save_pretrained` với `model.safetensors`; app hiện tại hỗ trợ trực tiếp định dạng này.
        Nguồn: [E5](https://huggingface.co/intfloat/multilingual-e5-base),
        [QASPER](https://aclanthology.org/2021.naacl-main.365/),
        [CachedMNRL](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).
        """),
    ]
    reranker = [
        md(
            "# 02 · Fine-tune reranker cho Research Desk\n"
            "Huấn luyện cross-encoder đa ngôn ngữ bằng BCE, chạy một GPU. "
            "Mặc định dùng BM25 để tạo candidate; có thể nạp E5 của notebook 01 cho hybrid search. "
            "Không cần API key hay source ZIP; chưa kèm kết quả GPU đã chạy."
        ),
        *common,
        md(
            "### 3. Cấu hình\nNếu OOM, giảm BATCH_SIZE còn 4 hoặc 2 và dùng RUN_NAME mới. "
            "Để dùng E5 vừa train, Add Input ZIP của notebook 01 rồi đặt RETRIEVER_ZIP; "
            "hoặc đặt RETRIEVER_PATH đến thư mục checkpoint đầy đủ."
        ),
        code(RERANK_CONFIG + TIME_CONFIG),
        md(DATA_NOTE),
        code(DATA),
        md("""
        ### 4. Nhãn âm và candidate validation
        QASPER không gắn nhãn mức liên quan cho mọi đoạn. Mặc định notebook lấy đoạn
        từ **paper train khác** (ưu tiên gần từ khóa) làm **nhãn âm yếu**: vẫn có thể
        có đoạn trả lời được câu hỏi. Không đánh dấu chúng là đã được người kiểm duyệt.
        File `negative_review.jsonl` cho phép người đọc sửa/loại nhãn sai, sau đó
        đặt `reviewed=true` ở từng dòng đã kiểm tra và dùng `REVIEWED_NEGATIVES`
        trong một experiment mới. Chỉ chấp nhận query/negative từ train, không trùng gold/held-out.

        Validation/test dùng candidate truy xuất thật, **không chèn gold** và không
        loại câu hỏi có candidate_recall=0. Vì vậy reranker không được tính công
        cho bằng chứng mà retriever chưa tìm thấy. Giữ CANDIDATES và retriever cố định
        khi so sánh reranker nền với fine-tuned. QASPER tiếng Anh chưa kiểm chứng truy vấn Việt–Anh.
        """),
        code(RERANK_PREPARE),
        md(
            "### 5. Fine-tune\nOptimizer AdamW + BCEWithLogitsLoss + fp16. "
            "Loại positive bị cắt đuôi; chọn checkpoint theo validation nDCG@10. "
            "Lưu loss từng epoch, nhãn yếu/đã review, seed, model revision và hash dữ liệu. "
            "Không tự resume run đã bị ngắt; hãy lưu Kaggle Version/Output và dùng RUN_NAME mới."
        ),
        code(RERANK_TRAIN),
        md(
            "### 6. Test cuối — mặc định tắt\nSau khi chốt cấu hình, bật RUN_FINAL_TEST. "
            "Đo retrieval-only, base reranker và fine-tuned reranker trên cùng candidate test."
        ),
        code(RERANK_TEST),
        md("### 7. Nạp lại offline trên CPU và tải checkpoint"),
        code(RERANK_RELOAD),
        export_cell("reranker"),
        md("""
        ### 8. Đưa vào hệ thống
        Tải `research-reranker-checkpoint.zip` và `research-reranker-experiment.zip`.
        Giải nén checkpoint vào `models/research-reranker`, sao cho config/tokenizer/weights
        nằm ngay trong thư mục này. Sửa **dòng cấu hình đang có**, tránh để hai dòng trùng tên:

        ```dotenv
        RA_RERANKER_PROVIDER=local
        RA_RERANKER_MODEL=models/research-reranker
        RA_RERANKER_DEVICE=cpu
        RA_RERANKER_BATCH_SIZE=2
        RA_RERANKER_MAX_LENGTH=512
        RA_RERANK_CANDIDATES=40
        ```

        Restart backend. Thay riêng reranker không cần reindex; thay E5 thì phải reindex.
        App dùng CrossEncoder.predict, tương thích với checkpoint được xuất.
        Không đổi weights thành `.pt` hoặc copy mỗi weights: cần cả tokenizer và cấu hình.
        Không mặc định triển khai bản fine-tuned nếu validation giảm; giữ base làm đối chứng.
        Nguồn: [model nền](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1),
        [reranker training](https://sbert.net/examples/cross_encoder/training/rerankers/README.html).
        """),
    ]
    for name, cells in (
        ("01_kaggle_finetune_e5.ipynb", e5),
        ("02_kaggle_finetune_reranker.ipynb", reranker),
    ):
        for i, value in enumerate(cells):
            value["id"] = f"research-{i:02d}"
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
        path = ROOT / "notebooks" / name
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    build()
