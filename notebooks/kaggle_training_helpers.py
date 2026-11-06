"""Standalone helpers embedded into the two Kaggle notebooks at build time.

Only standard-library imports run on import. Model/data downloads and GPU work are
explicit notebook cells. The generated notebooks do not import this local file.
"""

import hashlib
import importlib.metadata
import json
import math
import random
import re
import stat
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from research_assist.experiments.data import (
    file_hash,
    load_split,
    normalize,
    read_jsonl,
    validated_pairs,
)


class NotebookDeadlineReached(RuntimeError):
    """A cooperative deadline, raised between model batches rather than mid-write."""


class NotebookTimeBudget:
    def __init__(
        self,
        *,
        started=None,
        session_hours=12,
        train_stop_hours=10,
        export_reserve_hours=1,
        clock=None,
    ):
        self.clock = clock or time.monotonic
        self.started = self.clock() if started is None else started
        if not 0 < train_stop_hours < session_hours - export_reserve_hours < session_hours:
            raise ValueError("Require 0 < train stop < evaluation stop < session limit")
        self.session_hours = session_hours
        self.train_stop_hours = train_stop_hours
        self.export_reserve_hours = export_reserve_hours

    def training_expired(self):
        return self.clock() - self.started >= self.train_stop_hours * 3600

    def evaluation_expired(self):
        return (
            self.clock() - self.started >= (self.session_hours - self.export_reserve_hours) * 3600
        )

    def snapshot(self):
        elapsed = self.clock() - self.started
        return {
            "session_limit_hours": self.session_hours,
            "train_stop_after_hours": self.train_stop_hours,
            "evaluation_stop_after_hours": self.session_hours - self.export_reserve_hours,
            "export_reserve_hours": self.export_reserve_hours,
            "elapsed_hours": elapsed / 3600,
            "training_expired": self.training_expired(),
            "evaluation_expired": self.evaluation_expired(),
            "policy": "cooperative checks between batches; not a hard process timeout",
        }


def evaluate_with_budget(model, evaluate, budget=None):
    """Abort an incomplete evaluation at a model-batch boundary; never report partial metrics."""
    if budget is None:
        return evaluate()
    if budget.evaluation_expired():
        raise NotebookDeadlineReached("Evaluation budget exhausted; reserve time for export")
    if hasattr(model, "model"):  # CrossEncoder
        target = model.model
    else:  # SentenceTransformer: encode calls forward directly on the outer module.
        first = model[0]
        target = getattr(first, "auto_model", first)

    def check(module, inputs):
        if budget.evaluation_expired():
            raise NotebookDeadlineReached("Evaluation stopped at a batch boundary")

    handle = target.register_forward_pre_hook(check)
    try:
        return evaluate()
    finally:
        handle.remove()


def make_time_budget_callback(budget, checkpoint_minutes=15):
    from transformers import TrainerCallback

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self):
            self.last_save = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            now = time.monotonic()
            if now - self.last_save >= checkpoint_minutes * 60:
                control.should_save = True
                self.last_save = now
            if budget.training_expired():
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = False
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if budget.training_expired():
                control.should_training_stop = True
            if budget.evaluation_expired():
                control.should_evaluate = False
            return control

    return TimeBudgetCallback()


def download_qasper(destination, revision=""):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    destination = Path(destination)
    api = HfApi()
    revision = revision or api.dataset_info("allenai/qasper", revision="refs/convert/parquet").sha
    files = api.list_repo_files("allenai/qasper", repo_type="dataset", revision=revision)
    paths = {
        split: [
            hf_hub_download("allenai/qasper", name, repo_type="dataset", revision=revision)
            for name in files
            if name.endswith(".parquet") and f"/{split}/" in name
        ]
        for split in ("train", "validation", "test")
    }
    if any(not names for names in paths.values()):
        raise ValueError("Official QASPER parquet conversion is missing a split")
    dataset = load_dataset("parquet", data_files=paths)
    destination.mkdir(parents=True, exist_ok=True)
    for split in paths:
        rows = {row["id"]: row for row in dataset[split]}
        (destination / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    metadata = {"dataset": "allenai/qasper", "revision": revision, "license": "CC-BY-4.0"}
    write_json(destination / "download.json", metadata)
    return metadata


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "sentence-transformers",
            "transformers",
            "datasets",
            "accelerate",
            "huggingface-hub",
            "numpy",
            "pyarrow",
        )
    }


def unzip_checkpoint(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Choose an empty checkpoint extraction directory")
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe path or symlink inside checkpoint ZIP")
        archive.extractall(destination)
    if not (destination / "config.json").is_file():
        raise ValueError("ZIP must have config.json at its root, not inside another folder")
    return str(destination)


def export_checkpoint(model_dir, archive_path, experiment_dir, experiment_zip):
    """ZIP root is the deployable model folder; trainer state is a separate artifact."""
    model_dir, experiment_dir = Path(model_dir), Path(experiment_dir)
    if not (model_dir / "config.json").is_file():
        raise ValueError("Incomplete model: config.json missing")
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
    if not weights or not (model_dir / "research_model.json").is_file():
        raise ValueError("Incomplete model: weights or training manifest missing")
    paths = [
        p
        for p in sorted(model_dir.rglob("*"))
        if p.is_file()
        and "checkpoints" not in p.relative_to(model_dir).parts
        and p.name != "artifact_manifest.json"
    ]
    checksum = {
        "format_version": 1,
        "sha256": {p.relative_to(model_dir).as_posix(): file_hash(p) for p in paths},
    }
    write_json(model_dir / "artifact_manifest.json", checksum)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(model_dir).as_posix())
        archive.write(model_dir / "artifact_manifest.json", "artifact_manifest.json")
    with zipfile.ZipFile(experiment_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(experiment_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(experiment_dir).as_posix())
        for path in model_dir.glob("*.csv"):
            archive.write(path, "training/" + path.name)
        archive.write(model_dir / "research_model.json", "research_model.json")
    return str(archive_path), str(experiment_zip)


class NotebookBM25:
    """Inverted-index BM25; does not re-tokenize the corpus for every query."""

    def __init__(self, texts):
        self.lengths, self.postings = [], defaultdict(list)
        for i, text in enumerate(texts):
            counts = Counter(re.findall(r"\w+", text.casefold()))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.average = max(sum(self.lengths) / max(len(texts), 1), 1)

    def rank(self, query, count):
        scores = defaultdict(float)
        for term in set(re.findall(r"\w+", query.casefold())):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.lengths) - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, tf in postings:
                scores[i] += (
                    idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average))
                )
        return sorted(scores, key=lambda i: (-scores[i], i))[:count]


def build_candidates(
    data, split, *, count=40, retriever_path="", batch_size=16, device="cuda", time_budget=None
):
    """Use actual retrieval results, NEVER insert gold passages during evaluation."""
    import numpy as np

    corpus, queries = load_split(Path(data), split)
    index = NotebookBM25([c["text"] for c in corpus])
    matrix, query_vectors = None, None
    if retriever_path:
        import torch
        from sentence_transformers import SentenceTransformer

        retriever = SentenceTransformer(retriever_path, device=device, trust_remote_code=False)
        retriever.max_seq_length = 512
        matrix = evaluate_with_budget(
            retriever,
            lambda: retriever.encode(
                ["passage: " + c["text"] for c in corpus],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            time_budget,
        )
        query_vectors = evaluate_with_budget(
            retriever,
            lambda: retriever.encode(
                ["query: " + q["question"] for q in queries],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            time_budget,
        )
        del retriever
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    samples = []
    for qi, query in enumerate(queries):
        if time_budget and time_budget.evaluation_expired():
            raise NotebookDeadlineReached("Candidate generation stopped; no partial evaluation")
        lexical = index.rank(query["question"], max(100, count))
        if matrix is not None:
            values = matrix @ query_vectors[qi]
            take = min(max(100, count), len(corpus))
            dense = np.argpartition(-values, take - 1)[:take]
            dense = sorted(dense.tolist(), key=lambda i: (-float(values[i]), i))
            fused = defaultdict(float)
            for lane in (lexical, dense):
                for rank, i in enumerate(lane, 1):
                    fused[i] += 1 / (60 + rank)
            order = sorted(fused, key=lambda i: (-fused[i], i))[:count]
        else:
            order = lexical[:count]
        samples.append(
            {
                "query_id": query["id"],
                "query": query["question"],
                "relevant_ids": query["relevant_ids"],
                "documents": [{"id": corpus[i]["id"], "text": corpus[i]["text"]} for i in order],
            }
        )
    return samples


def rank_metrics(ids, gold, k=10):
    ids = list(dict.fromkeys(ids))[:k]
    hits = [rank for rank, cid in enumerate(ids, 1) if cid in gold]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return {
        f"recall@{k}": len(hits) / len(gold),
        f"mrr@{k}": 1 / hits[0] if hits else 0.0,
        f"ndcg@{k}": sum(1 / math.log2(i + 1) for i in hits) / ideal,
    }


def evaluate_candidates(model, samples, batch_size=8):
    """Include queries whose gold is completely absent from the candidate list."""
    totals, rows = defaultdict(list), []
    for sample in samples:
        docs = sample["documents"]
        started = time.perf_counter()
        if model is None:
            order = list(range(len(docs)))
        elif docs:
            scores = model.predict(
                [(sample["query"], d["text"]) for d in docs],
                batch_size=batch_size,
                show_progress_bar=False,
            )
            if len(scores) != len(docs) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("Invalid reranker scores")
            order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), docs[i]["id"]))
        else:
            order = []
        ranked = [docs[i]["id"] for i in order]
        gold = set(sample["relevant_ids"])
        metrics = rank_metrics(ranked, gold)
        metrics["candidate_recall"] = len(gold.intersection(d["id"] for d in docs)) / len(gold)
        for name, value in metrics.items():
            totals[name].append(value)
        rows.append(
            {
                "query_id": sample["query_id"],
                "ranked_ids": ranked[:10],
                "metrics": metrics,
                "rerank_ms": 1000 * (time.perf_counter() - started),
            }
        )
    if not rows:
        raise ValueError("No evaluation queries")
    return {
        "metrics": {k: sum(v) / len(v) for k, v in totals.items()},
        "queries": len(rows),
        "per_query": rows,
        "gold_injection": False,
        "scope": "all paragraphs in the selected official split",
    }


def make_reranker_pairs(data, *, negatives_per_query=3, seed=42, reviewed_path="", synthetic=None):
    """Default negatives are cross-paper WEAK labels, never claimed to be human-reviewed.

    Reviewed mode accepts JSONL: query_id, negative_id, label=0, reviewed=true.
    All provenance is checked again; held-out text is never used for training.
    """
    if negatives_per_query < 1:
        raise ValueError("negatives_per_query must be positive")
    data = Path(data)
    all_corpus = read_jsonl(data / "corpus.jsonl")
    heldout = {normalize(c["text"]) for c in all_corpus if c["split"] != "train"}
    corpus = [
        c for c in all_corpus if c["split"] == "train" and normalize(c["text"]) not in heldout
    ]
    by_id = {c["id"]: c for c in corpus}
    grouped = {}
    for p in validated_pairs(data, Path(synthetic) if synthetic else None):
        qid = (
            p.get("query_id") or "custom_" + hashlib.sha256(p["question"].encode()).hexdigest()[:16]
        )
        key = (qid, p["question"], p["paper_id"])
        grouped.setdefault(key, set()).add(p["positive_id"])
    reviewed = defaultdict(list)
    valid_qids = {key[0] for key in grouped}
    if reviewed_path:
        for row in read_jsonl(Path(reviewed_path)):
            if row.get("reviewed") is not True or row.get("label") != 0:
                raise ValueError("Reviewed negative requires reviewed=true and label=0")
            if row.get("query_id") not in valid_qids or row.get("negative_id") not in by_id:
                raise ValueError(
                    "Reviewed negative has unknown query or held-out/duplicate passage"
                )
            reviewed[row["query_id"]].append(row["negative_id"])
    lexical = NotebookBM25([c["text"] for c in corpus])
    rng = random.Random(seed)
    rows, review_rows = [], []
    for (qid, question, paper_id), gold in sorted(grouped.items()):
        gold_texts = {normalize(by_id[cid]["text"]) for cid in gold}
        if reviewed_path:
            chosen = list(dict.fromkeys(reviewed[qid]))[:negatives_per_query]
            if any(cid in gold or normalize(by_id[cid]["text"]) in gold_texts for cid in chosen):
                raise ValueError("A reviewed negative is actually a known positive")
        else:
            ranked = lexical.rank(question, 200)
            selected = set(ranked)
            remainder = [i for i in range(len(corpus)) if i not in selected]
            rng.shuffle(remainder)
            chosen = []
            for i in [*ranked, *remainder]:
                c = corpus[i]
                if c["paper_id"] != paper_id and normalize(c["text"]) not in gold_texts:
                    chosen.append(c["id"])
                    if len(chosen) == negatives_per_query:
                        break
        if not chosen:
            continue
        for cid in sorted(gold):
            rows.append(
                {
                    "query_id": qid,
                    "query": question,
                    "passage_id": cid,
                    "passage": by_id[cid]["text"],
                    "label": 1.0,
                    "source": "reviewed_positive",
                }
            )
        for cid in chosen:
            rows.append(
                {
                    "query_id": qid,
                    "query": question,
                    "passage_id": cid,
                    "passage": by_id[cid]["text"],
                    "label": 0.0,
                    "source": "human_reviewed" if reviewed_path else "cross_paper_weak",
                }
            )
            review_rows.append(
                {
                    "query_id": qid,
                    "question": question,
                    "negative_id": cid,
                    "passage": by_id[cid]["text"],
                    "label": 0,
                    "reviewed": bool(reviewed_path),
                }
            )
    if not rows or not any(r["label"] == 0 for r in rows):
        raise ValueError("No usable negatives; supply reviewed negatives from training papers")
    return rows, review_rows


def train_reranker(
    base,
    output,
    rows,
    validation_samples,
    *,
    revision="",
    epochs=2,
    batch_size=8,
    accumulation=2,
    learning_rate=2e-5,
    max_length=512,
    seed=42,
    device="cuda",
    time_budget=None,
    checkpoint_minutes=15,
):
    """Real PyTorch BCE optimization; choose checkpoint by validation nDCG@10."""
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup, set_seed

    if epochs < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError("epochs, batch_size and accumulation must be positive integers")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; enable Kaggle GPU")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; use a new RUN_NAME")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    options = {"device": device, "max_length": max_length, "trust_remote_code": False}
    if revision:
        options["revision"] = revision
    model = CrossEncoder(base, **options)
    if model.model.config.num_labels != 1:
        raise ValueError("Reranker must have one relevance logit")
    original_count = len(rows)
    # Positive evidence truncated at the answer tail must not silently retain label=1.
    rows = [
        r
        for r in rows
        if r["label"] == 0
        or len(model.tokenizer(r["query"], r["passage"], truncation=False)["input_ids"])
        <= max_length
    ]
    positives = sum(r["label"] == 1 for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError("Need both non-truncated positives and negatives")

    def collate(batch):
        features = model.tokenizer(
            [r["query"] for r in batch],
            [r["passage"] for r in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return features, torch.tensor([r["label"] for r in batch], dtype=torch.float32)

    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    before = None
    try:
        before = evaluate_with_budget(
            model, lambda: evaluate_candidates(model, validation_samples, batch_size), time_budget
        )
    except NotebookDeadlineReached:
        print("Baseline validation skipped/incomplete due to the session deadline.", flush=True)
    write_json(output / "validation_before.json", before)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate, weight_decay=0.01)
    updates = math.ceil(len(loader) / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * 0.1), updates)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    best, best_epoch, history = -math.inf, 0, []
    optimizer_steps, stopped = 0, False
    last_save = time.monotonic()
    recovery = output.parent / "recovery-reranker"
    for epoch in range(epochs):
        model.model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (features, labels) in enumerate(loader):
            if time_budget and time_budget.training_expired():
                stopped = True
                break
            features = {key: value.to(device) for key, value in features.items()}
            labels = labels.to(device)
            # Scale the final partial accumulation group correctly.
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model.model(**features).logits.reshape(-1)
                raw_loss = loss_fn(logits.float(), labels)
                loss = raw_loss / group_size
            if not torch.isfinite(raw_loss):
                raise ValueError("Non-finite training loss; reduce learning rate or disable AMP")
            scaler.scale(loss).backward()
            losses.append(float(raw_loss.detach().cpu()))
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if time.monotonic() - last_save >= checkpoint_minutes * 60:
                    model.save_pretrained(str(recovery), safe_serialization=True)
                    write_json(
                        recovery / "checkpoint_status.json",
                        {
                            "selection": "last_unvalidated",
                            "optimizer_steps": optimizer_steps,
                            "epoch": epoch + (step + 1) / len(loader),
                            "resume_optimizer_supported": False,
                        },
                    )
                    last_save = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        model.model.eval()
        validation = None
        try:
            validation = evaluate_with_budget(
                model,
                lambda: evaluate_candidates(model, validation_samples, batch_size),
                time_budget,
            )
        except NotebookDeadlineReached:
            print("Validation stopped/incomplete; no partial metric is reported.", flush=True)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "validation": validation["metrics"] if validation else None,
            "partial_epoch": stopped,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation and optimizer_steps and validation["metrics"]["ndcg@10"] > best:
            best, best_epoch = validation["metrics"]["ndcg@10"], epoch + 1
            model.save_pretrained(str(output), safe_serialization=True)
            write_json(output / "validation_after.json", validation)
        # Kaggle Saved Version / Output can preserve this between sessions.
        # Notebook deliberately does not silently resume a different experiment.
        write_json(output / "training_history.json", history)
        if stopped or (time_budget and time_budget.training_expired()):
            stopped = True
            break
    selection = "best_validation_checkpoint" if best_epoch else "last_unvalidated"
    if not best_epoch:
        model.save_pretrained(str(output), safe_serialization=True)
        write_json(output / "validation_after.json", None)
    if not optimizer_steps:
        selection = "base_not_trained"
    manifest = {
        "format_version": 2,
        "model_family": "cross_encoder",
        "fine_tuned": optimizer_steps > 0,
        "base_model": base,
        "revision": revision,
        "best_epoch": best_epoch,
        "loss": "BCEWithLogitsLoss",
        "max_seq_length": max_length,
        "rows_used": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "rows_excluded_truncation": original_count - len(rows),
        "negative_sources": dict(Counter(r["source"] for r in rows if r["label"] == 0)),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "accumulation": accumulation,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
        },
        "validation_before": before["metrics"] if before else None,
        "validation_after": json.loads((output / "validation_after.json").read_text()),
        "optimizer_steps": optimizer_steps,
        "selection": selection,
        "stop_reason": "time_budget" if stopped else "epochs_completed",
        "time_budget": time_budget.snapshot() if time_budget else None,
        "history": history,
        "versions": package_versions(),
    }
    if manifest["validation_after"]:
        manifest["validation_after"] = manifest["validation_after"]["metrics"]
    write_json(output / "research_model.json", manifest)
    return manifest
