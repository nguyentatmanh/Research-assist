"""SWIFT 3.12.3 QLoRA recipe and bounded Kaggle orchestration, embedded in notebook 06."""

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

QWEN_REQUIREMENTS = [
    "ms-swift==3.12.3",
    "transformers==4.57.1",
    "peft==0.17.1",
    "trl==0.23.1",
    "datasets==3.6.0",
    "accelerate==1.11.0",
    "bitsandbytes==0.48.2",
    "huggingface-hub==0.36.0",
    "sentence-transformers==5.1.2",
    "numpy==2.2.6",
    "pandas==2.2.3",
    "pyarrow==20.0.0",
    "scikit-learn==1.6.1",
]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def session_start(clock_path, elapsed_hours=0):
    """Retain the oldest observed start across cell reruns and kernel restarts."""
    now = time.time()
    try:
        import psutil

        kernel_start = psutil.Process(os.getpid()).create_time()
    except ImportError:
        kernel_start = now
    started = min(kernel_start, now - elapsed_hours * 3600)
    path = Path(clock_path)
    if path.exists():
        started = min(started, json.loads(path.read_text())["started_epoch"])
    atomic_json(path, {"started_epoch": started})
    return started


def install_qwen_environment(python, requirements, constraints, deadline):
    remaining = min(1800, deadline - time.time())
    if remaining <= 0:
        raise RuntimeError("No setup budget remaining.")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--prefer-binary",
            "--only-binary=numpy,pandas,pyarrow,scikit-learn",
            "--upgrade-strategy",
            "only-if-needed",
            "-c",
            str(constraints),
            *requirements,
        ],
        check=True,
        timeout=remaining,
    )


def make_triplets(data, tokenizer, instruction, max_length=512, seed=42, limit=6000):
    """QASPER paper splits, all known gold excluded from lexical negative candidates."""
    # These helpers are embedded from the repository's existing QASPER data preparation.
    from research_assist.experiments.data import load_split, normalize, read_jsonl, validated_pairs

    corpus, train_queries = load_split(Path(data), "train")
    positives = validated_pairs(Path(data))
    excluded_texts = {
        normalize(item["text"])
        for item in read_jsonl(Path(data) / "corpus.jsonl")
        if item["split"] != "train"
    }
    by_paper = {}
    gold = {}
    for query in train_queries:
        gold.setdefault(query["question"], set()).update(query["relevant_ids"])
    for row in corpus:
        by_paper.setdefault(row["paper_id"], []).append(row)
    rng = random.Random(seed)
    rng.shuffle(positives)
    train, dropped = [], 0

    def fits(text):
        return len(tokenizer.encode(text, add_special_tokens=False)) + 1 <= max_length

    def row(question, positive, negative):
        return {
            "messages": [{"role": "user", "content": f"Instruct: {instruction}\nQuery:{question}"}],
            "positive_messages": [[{"role": "user", "content": positive}]],
            "negative_messages": [[{"role": "user", "content": negative}]],
        }

    def negative_for(question, paper, forbidden, positive):
        terms = set(question.casefold().split())
        candidates = [
            item
            for item in by_paper.get(paper, [])
            if item["id"] not in forbidden
            and normalize(item["text"]) != normalize(positive)
            and normalize(item["text"]) not in excluded_texts
            and fits(item["text"])
        ]
        # Lexical candidates are silver negatives: QASPER gold may be incomplete.
        candidates.sort(
            key=lambda item: (-len(terms & set(item["text"].casefold().split())), item["id"])
        )
        return candidates[0]["text"] if candidates else None

    for item in positives:
        question, positive = item["question"], item["positive"]
        query = f"Instruct: {instruction}\nQuery:{question}"
        negative = negative_for(
            question, item["paper_id"], gold.get(question, set()) | {item["positive_id"]}, positive
        )
        if not negative or not fits(query) or not fits(positive):
            dropped += 1
            continue
        train.append(row(question, positive, negative))
        if len(train) >= limit:
            break
    validation_corpus, queries = load_split(Path(data), "validation")
    excluded_texts = set()
    by_paper.clear()
    lookup = {item["id"]: item["text"] for item in validation_corpus}
    for item in validation_corpus:
        by_paper.setdefault(item["paper_id"], []).append(item)
    validation = []
    for query in sorted(queries, key=lambda q: q["id"]):
        positive = lookup[query["relevant_ids"][0]]
        negative = negative_for(
            query["question"], query["paper_id"], set(query["relevant_ids"]), positive
        )
        if (
            negative
            and fits(positive)
            and fits(f"Instruct: {instruction}\nQuery:{query['question']}")
        ):
            validation.append(row(query["question"], positive, negative))
        if len(validation) >= 64:
            break
    if len(train) < 16 or len(validation) < 4:
        raise ValueError("Too few non-truncated triplets. Check data or increase MAX_LENGTH.")
    return (
        train,
        validation,
        {
            "train": len(train),
            "validation": len(validation),
            "excluded": dropped,
            "negative_policy": "same-paper lexical silver",
        },
    )


def prepare_worker(config):
    import importlib.metadata

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    from notebooks.kaggle_training_helpers import download_qasper
    from research_assist.experiments.data import load_split, prepare_qasper, read_jsonl, write_jsonl

    run = Path(config["run"])
    revision = config.get("base_revision")
    if config.get("resume"):
        recipe = json.loads((Path(config["resume"]).parent / "resume_recipe.json").read_text())
        for name in ("base_model", "instruction", "max_length", "learning_rate"):
            if recipe[name] != config[name]:
                raise ValueError(f"Resume recipe differs: {name}. Keep original configuration.")
        if revision and revision != recipe["base_revision"]:
            raise ValueError("Resume must use the original base revision.")
        revision = recipe["base_revision"]
    revision = revision or HfApi().model_info(config["base_model"]).sha
    config["base_revision"] = revision
    config["base_path"] = snapshot_download(config["base_model"], revision=revision)
    data = Path(config.get("prepared_data") or "/kaggle/temp/qwen-qasper")
    if not config.get("prepared_data"):
        raw = Path("/kaggle/temp/qwen-qasper-raw")
        metadata = download_qasper(
            raw, recipe.get("dataset_revision", "") if config.get("resume") else ""
        )
        config["dataset_revision"] = metadata["revision"]
        manifest = prepare_qasper(raw, data)
        atomic_json(run / "qasper_download.json", metadata)
        atomic_json(run / "dataset_manifest.json", manifest)
    paper_splits, ids = {}, set()
    for row in read_jsonl(data / "corpus.jsonl"):
        if row["id"] in ids or paper_splits.get(row["paper_id"], row["split"]) != row["split"]:
            raise ValueError("Duplicate corpus ID or paper leaking across splits.")
        ids.add(row["id"])
        paper_splits[row["paper_id"]] = row["split"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_path"], trust_remote_code=False)
    train, validation, counts = make_triplets(
        data, tokenizer, config["instruction"], config["max_length"], limit=config["pair_limit"]
    )
    config["train_data"], config["validation_data"] = (
        str(run / "train.jsonl"),
        str(run / "validation.jsonl"),
    )
    write_jsonl(Path(config["train_data"]), train)
    write_jsonl(Path(config["validation_data"]), validation)
    config["data_sha256"] = {
        name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
        for name in ("train_data", "validation_data")
    }
    if config.get("resume") and config["data_sha256"] != recipe["data_sha256"]:
        raise ValueError("Resume data differs from original training run.")
    corpus, queries = load_split(data, "validation")
    chosen_queries = sorted(queries, key=lambda item: item["id"])[:16]
    gold_ids = {cid for query in chosen_queries for cid in query["relevant_ids"]}
    selected_docs = [item for item in corpus if item["id"] in gold_ids]
    selected_docs += [item for item in corpus if item["id"] not in gold_ids][
        : max(0, 256 - len(selected_docs))
    ]
    positions = {item["id"]: i for i, item in enumerate(selected_docs)}
    atomic_json(
        run / "evaluation_panel.json",
        {
            "documents": [item["text"] for item in selected_docs],
            "questions": [item["question"] for item in chosen_queries],
            "gold": [[positions[cid] for cid in item["relevant_ids"]] for item in chosen_queries],
        },
    )
    config["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "ms-swift", "transformers", "peft", "trl", "bitsandbytes", "datasets")
    }
    atomic_json(run / "config.json", config)
    atomic_json(run / "triplet_statistics.json", counts)


def swift_arguments(config):
    """Flags verified against the pinned SWIFT wheel, not a moving latest API."""
    result = [
        "--model",
        config["base_path"],
        "--model_type",
        "qwen3_emb",
        "--task_type",
        "embedding",
        "--train_type",
        "lora",
        "--tuner_backend",
        "peft",
        "--use_hf",
        "true",
        "--quant_method",
        "bnb",
        "--quant_bits",
        "4",
        "--bnb_4bit_compute_dtype",
        "float16",
        "--bnb_4bit_quant_type",
        "nf4",
        "--bnb_4bit_use_double_quant",
        "true",
        "--torch_dtype",
        "float16",
        "--fp16",
        "true",
        "--bf16",
        "false",
        "--attn_impl",
        "sdpa",
        "--target_modules",
        "all-linear",
        "--lora_rank",
        "8",
        "--lora_alpha",
        "16",
        "--lora_dropout",
        "0.05",
        "--gradient_checkpointing",
        "true",
        "--dataset",
        config["train_data"],
        "--val_dataset",
        config["validation_data"],
        "--split_dataset_ratio",
        "0",
        "--max_length",
        str(config["max_length"]),
        "--truncation_strategy",
        "delete",
        "--loss_type",
        "infonce",
        "--label_names",
        "labels",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        str(config["learning_rate"]),
        "--num_train_epochs",
        "1",
        "--max_steps",
        str(config["max_steps"]),
        "--warmup_ratio",
        "0.05",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "100",
        "--save_strategy",
        "steps",
        "--save_steps",
        "100",
        "--save_total_limit",
        "2",
        "--save_only_model",
        "false",
        "--load_best_model_at_end",
        "false",
        "--metric_for_best_model",
        "eval_loss",
        "--greater_is_better",
        "false",
        "--logging_steps",
        "5",
        "--report_to",
        "none",
        "--dataloader_num_workers",
        "0",
        "--dataset_num_proc",
        "1",
        "--dataloader_drop_last",
        "true",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--output_dir",
        config["output_dir"],
        "--add_version",
        "false",
    ]
    if config.get("resume"):
        result += ["--resume_from_checkpoint", config["resume"]]
    return result


def deadline_callback(deadline, progress_path, checkpoint_seconds=900, clock=time.time):
    from transformers import TrainerCallback

    class DeadlineCallback(TrainerCallback):
        def __init__(self):
            self.last_save = clock()

        @staticmethod
        def signature(model):
            import torch

            digest, count = hashlib.sha256(), 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                value = parameter.detach().float().cpu()
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        "Non-finite trainable weights; checkpoint is not deployable."
                    )
                digest.update(name.encode())
                digest.update(value.numpy().tobytes())
                count += value.numel()
            if not count:
                raise RuntimeError("No trainable adapter parameters found.")
            return digest.hexdigest()

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.initial_signature = self.signature(model)

        def on_step_end(self, args, state, control, **kwargs):
            now = clock()
            atomic_json(progress_path, {"optimizer_steps": state.global_step, "updated": now})
            if now >= deadline:
                control.should_training_stop = True
                control.should_save = state.global_step > 0
                control.should_evaluate = False
            elif now - self.last_save >= checkpoint_seconds:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            self.last_save = clock()
            signature = self.signature(kwargs["model"])
            atomic_json(
                Path(progress_path).parent / "training_integrity.json",
                {
                    "optimizer_steps": state.global_step,
                    "adapter_updated": signature != self.initial_signature,
                },
            )

    return DeadlineCallback()


def train_worker(config):
    # Never use batch=1 without explicit negatives. Accumulation does not enlarge InfoNCE batches.
    os.environ["INFONCE_USE_BATCH"] = "false"
    os.environ["INFONCE_TEMPERATURE"] = "0.05"
    os.environ["INFONCE_HARD_NEGATIVES"] = "1"
    from swift.llm import sft_main
    from swift.plugin import extra_callbacks

    if time.time() >= config["train_deadline"]:
        raise RuntimeError("Training deadline reached before loading model.")
    extra_callbacks.append(
        deadline_callback(config["train_deadline"], Path(config["run"]) / "progress.json")
    )
    result = sft_main(swift_arguments(config))
    atomic_json(Path(config["run"]) / "train_result.json", result)


def run_bounded(command, log_path, deadline, env=None):
    """A hung setup/evaluation cannot consume the entire 12h session."""
    if time.time() >= deadline:
        return {"returncode": None, "timed_out": True}
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        last_notice = time.time()
        try:
            while process.poll() is None:
                if time.time() - last_notice >= 60:
                    remaining = max(0, deadline - time.time()) / 3600
                    print(
                        f"{Path(log_path).name}: running; budget remaining {remaining:.2f}h",
                        flush=True,
                    )
                    last_notice = time.time()
                if time.time() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                    return {"returncode": process.returncode, "timed_out": True}
                time.sleep(2)
            return {"returncode": process.returncode, "timed_out": False}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)


def complete_checkpoints(output_dir):
    """Ignore incomplete writes after a hard timeout. Full checkpoints retain resume state."""
    result = []
    for path in Path(output_dir).rglob("checkpoint-*"):
        required = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        )
        if not path.is_dir() or not all((path / name).is_file() for name in required):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            if state["global_step"] > 0:
                result.append((state["global_step"], path, state))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result)


def export_adapter(config, process_status):
    integrity_path = Path(config["run"]) / "training_integrity.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    if not integrity.get("adapter_updated"):
        raise RuntimeError("No verified adapter weight update; inspect training_integrity.json.")
    checkpoints = [
        item
        for item in complete_checkpoints(config["output_dir"])
        if item[0] <= integrity.get("optimizer_steps", 0)
    ]
    if not checkpoints:
        raise RuntimeError(
            "No complete trained checkpoint. Inspect train.log; no base model is exported."
        )
    _, latest, state = checkpoints[-1]
    # State may record a best checkpoint from the original session path after resume.
    best_name = Path(state.get("best_model_checkpoint") or "").name
    chosen = next((item for item in checkpoints if item[1].name == best_name), checkpoints[-1])
    step, selected, _ = chosen
    output = Path(config["run"]) / "qwen3-adapter"
    output.mkdir(exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(selected / name, output / name)
    manifest = {
        "format": "swift_qwen3_adapter_v1",
        "model_family": "qwen3",
        "fine_tuned": True,
        "optimizer_steps": step,
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "query_instruction": config["instruction"],
        "max_seq_length": config["max_length"],
        "suffix": "<|endoftext|>",
        "loss": "infonce",
        "infonce": {"temperature": 0.05, "use_batch": False, "hard_negatives": 1},
        "quantization": "bnb NF4 double quantization, FP16 compute",
        "negative_policy": "same-paper lexical silver",
        "selection": "best_eval_loss" if selected.name == best_name else "last_unvalidated",
        "process_status": process_status,
        "packages": config.get("packages", {}),
        "data_sha256": config["data_sha256"],
    }
    atomic_json(output / "research_model.json", manifest)
    # Package the last checkpoint too, including optimizer/RNG, separately from the deploy adapter.
    resume = Path(config["run"]) / "resume"
    if not resume.exists():
        shutil.copytree(latest, resume)
    atomic_json(
        Path(config["run"]) / "resume_recipe.json",
        {
            "dataset_revision": config.get("dataset_revision", ""),
            **{
                key: config[key]
                for key in (
                    "base_model",
                    "base_revision",
                    "instruction",
                    "max_length",
                    "learning_rate",
                    "data_sha256",
                )
            },
        },
    )
    return output


def evaluate_worker(config):
    import numpy as np

    from notebooks.qwen_adapter_encoder import QwenAdapterEncoder

    run = Path(config["run"])
    panel = json.loads((run / "evaluation_panel.json").read_text(encoding="utf-8"))
    encoder = QwenAdapterEncoder(
        config["base_path"],
        config["instruction"],
        config["max_length"],
        adapter_path=run / "qwen3-adapter",
    )

    def measure():
        vectors = []
        for text in panel["documents"]:
            if time.time() >= config["eval_deadline"]:
                raise TimeoutError("Evaluation reserve reached; no partial metrics reported.")
            vectors.append(encoder.encode_document([text], batch_size=1)[0])
        corpus = np.asarray(vectors)
        recall, reciprocal = [], []
        for query, gold in zip(panel["questions"], panel["gold"], strict=True):
            if time.time() >= config["eval_deadline"]:
                raise TimeoutError("Evaluation reserve reached; no partial metrics reported.")
            query_vector = encoder.encode_query([query], batch_size=1)[0]
            ranking = np.argsort(-(corpus @ query_vector)).tolist()
            recall.append(len(set(ranking[:10]) & set(gold)) / len(gold))
            reciprocal.append(
                next((1 / (i + 1) for i, item in enumerate(ranking[:10]) if item in gold), 0)
            )
        return {"recall@10": float(np.mean(recall)), "mrr@10": float(np.mean(reciprocal))}

    with encoder.model.disable_adapter():
        before = measure()
    after = measure()
    atomic_json(
        run / "retrieval_validation.json",
        {
            "baseline": before,
            "fine_tuned": after,
            "queries": len(panel["questions"]),
            "corpus": len(panel["documents"]),
            "scope": "fixed validation subset, not full benchmark",
            "quantization": "same NF4 base for both evaluations",
            "test_set_used": False,
        },
    )
