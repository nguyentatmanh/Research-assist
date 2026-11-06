"""Pinned SWIFT generative-reranker/BCE recipe for Qwen3-Reranker-4B on Kaggle."""

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from notebooks.kaggle_qwen_training import (
    atomic_json,
    complete_checkpoints,
    deadline_callback,
    swift_arguments,
)
from notebooks.kaggle_training_helpers import NotebookBM25, download_qasper
from notebooks.qwen_reranker_scorer import QwenRerankerScorer, pair_token_ids
from research_assist.experiments.data import (
    load_split,
    normalize,
    prepare_qasper,
    read_jsonl,
    validated_pairs,
    write_jsonl,
)

RECIPE = "qwen3-reranker-bce-v1"


def reranker_arguments(config):
    options = dict(zip(swift_arguments(config)[::2], swift_arguments(config)[1::2], strict=True))
    options.update(
        {
            "--model_type": "qwen3_reranker",
            "--task_type": "generative_reranker",
            "--loss_type": "generative_reranker",
        }
    )
    return [item for pair in options.items() for item in pair]


def set_reranker_environment():
    # One input group expands to exactly two query/document pairs on each device.
    os.environ["MAX_POSITIVE_SAMPLES"] = "1"
    os.environ["MAX_NEGATIVE_SAMPLES"] = "1"
    os.environ["GENERATIVE_RERANKER_POSITIVE_TOKEN"] = "yes"
    os.environ["GENERATIVE_RERANKER_NEGATIVE_TOKEN"] = "no"


def reranker_row(question, positive, negative, instruction):
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ],
        "positive_messages": [[{"role": "assistant", "content": positive}]],
        "negative_messages": [[{"role": "assistant", "content": negative}]],
    }


def verify_swift_prompt(base_path, row, instruction, max_length):
    """Fail early if the pinned SWIFT training template differs from inference token IDs."""
    from swift.llm import get_model_tokenizer, get_template

    _, processor = get_model_tokenizer(
        str(base_path),
        model_type="qwen3_reranker",
        task_type="generative_reranker",
        load_model=False,
        use_hf=True,
    )
    template = get_template(
        "qwen3_reranker", processor, max_length=max_length, truncation_strategy="delete"
    )
    template.set_mode("train")
    encoded = template.encode(row)
    question = row["messages"][-1]["content"]
    expected = [
        pair_token_ids(processor, question, row[key][0][0]["content"], instruction, max_length)
        for key in ("positive_messages", "negative_messages")
    ]
    if encoded["input_ids"] != expected or encoded["labels"] != [1, 0]:
        raise RuntimeError("SWIFT train/inference prompt mismatch; refusing to train.")
    return {
        "matched": True,
        "labels": encoded["labels"],
        "sequence_lengths": [len(ids) for ids in expected],
    }


def prepare_reranker_data(data, tokenizer, instruction, max_length=512, limit=6000):
    data = Path(data)
    all_corpus = read_jsonl(data / "corpus.jsonl")
    seen_ids, paper_splits = set(), {}
    for item in all_corpus:
        if (
            item["id"] in seen_ids
            or paper_splits.get(item["paper_id"], item["split"]) != item["split"]
        ):
            raise ValueError("Duplicate corpus ID or paper leakage between splits.")
        seen_ids.add(item["id"])
        paper_splits[item["paper_id"]] = item["split"]
    heldout = {normalize(item["text"]) for item in all_corpus if item["split"] != "train"}
    pairs = validated_pairs(data)
    train_corpus, train_queries = load_split(data, "train")
    val_corpus, val_queries = load_split(data, "validation")

    def fits(question, document):
        try:
            pair_token_ids(tokenizer, question, document, instruction, max_length)
            return True
        except ValueError:
            return False

    def groups_for(corpus, queries, examples, count, split):
        lookup = {item["id"]: item for item in corpus}
        by_paper, gold = defaultdict(list), defaultdict(set)
        for item in corpus:
            by_paper[item["paper_id"]].append(item)
        for query in queries:
            gold[query["question"]].update(query["relevant_ids"])
        indexes = {
            paper: NotebookBM25([item["text"] for item in rows]) for paper, rows in by_paper.items()
        }
        result, provenance, excluded = [], [], 0
        for item in examples:
            question, cid = item["question"], item["positive_id"]
            positive = lookup[cid]
            forbidden = {normalize(lookup[i]["text"]) for i in gold[question]}
            if not fits(question, positive["text"]):
                excluded += 1
                continue
            rows = by_paper[item["paper_id"]]
            candidates = indexes[item["paper_id"]].rank(question, len(rows))
            candidates += [i for i in range(len(rows)) if i not in set(candidates)]
            negative = next(
                (
                    rows[i]
                    for i in candidates
                    if normalize(rows[i]["text"]) not in forbidden
                    and (split != "train" or normalize(rows[i]["text"]) not in heldout)
                    and fits(question, rows[i]["text"])
                ),
                None,
            )
            if negative is None:
                excluded += 1
                continue
            result.append(reranker_row(question, positive["text"], negative["text"], instruction))
            provenance.append(
                {
                    "split": split,
                    "query_id": item.get("query_id"),
                    "paper_id": item["paper_id"],
                    "positive_id": cid,
                    "negative_id": negative["id"],
                    "negative_label": "silver",
                }
            )
            if len(result) >= count:
                break
        return result, provenance, excluded

    random.Random(42).shuffle(pairs)
    train, train_provenance, train_excluded = groups_for(
        train_corpus, train_queries, pairs, limit, "train"
    )
    val_examples = [
        {
            "query_id": q["id"],
            "question": q["question"],
            "positive_id": q["relevant_ids"][0],
            "paper_id": q["paper_id"],
        }
        for q in sorted(val_queries, key=lambda q: q["id"])
    ]
    validation, val_provenance, val_excluded = groups_for(
        val_corpus, val_queries, val_examples, 64, "validation"
    )
    if len(train) < 16 or len(validation) < 4:
        raise ValueError("Too few complete query/document pairs within MAX_LENGTH.")

    index = NotebookBM25([item["text"] for item in val_corpus])
    lookup = {item["id"]: item for item in val_corpus}
    panel = []
    for query in sorted(val_queries, key=lambda q: q["id"]):
        if not all(fits(query["question"], lookup[cid]["text"]) for cid in query["relevant_ids"]):
            continue
        ranked = index.rank(query["question"], 100)
        candidates = [
            val_corpus[i] for i in ranked if fits(query["question"], val_corpus[i]["text"])
        ][:20]
        if not candidates:
            continue
        # Never insert gold: missed evidence remains a retrieval failure in the metrics.
        panel.append(
            {
                "id": query["id"],
                "question": query["question"],
                "relevant_ids": query["relevant_ids"],
                "candidates": candidates,
            }
        )
        if len(panel) >= 16:
            break
    if not panel:
        raise ValueError("No eligible validation queries for reranking evaluation.")
    return (
        train,
        validation,
        panel,
        {
            "train_groups": len(train),
            "train_pairs": 2 * len(train),
            "validation_groups": len(validation),
            "validation_pairs": 2 * len(validation),
            "excluded_train_groups": train_excluded,
            "excluded_validation_groups": val_excluded,
            "evaluation_queries": len(panel),
            "negative_policy": "same-paper BM25 silver",
            "provenance": train_provenance + val_provenance,
        },
    )


def prepare_reranker_worker(config):
    import importlib.metadata

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    run = Path(config["run"])
    if config["base_model"] != "Qwen/Qwen3-Reranker-4B":
        raise ValueError("This recipe trains Qwen3-Reranker-4B only.")
    recipe = {}
    if config.get("resume"):
        recipe = json.loads((Path(config["resume"]).parent / "resume_recipe.json").read_text())
        for key in ("recipe", "base_model", "instruction", "max_length", "learning_rate"):
            if recipe.get(key) != config[key]:
                raise ValueError(f"Resume recipe mismatch: {key}.")
    config["base_revision"] = (
        recipe.get("base_revision") or HfApi().model_info(config["base_model"]).sha
    )
    config["base_path"] = snapshot_download(config["base_model"], revision=config["base_revision"])
    data = Path(config.get("prepared_data") or "/kaggle/temp/qwen-reranker-data")
    if not config.get("prepared_data"):
        raw = Path("/kaggle/temp/qwen-reranker-raw")
        metadata = download_qasper(raw, recipe.get("dataset_revision", ""))
        config["dataset_revision"] = metadata["revision"]
        atomic_json(run / "dataset_download.json", metadata)
        atomic_json(run / "dataset_manifest.json", prepare_qasper(raw, data))
    tokenizer = AutoTokenizer.from_pretrained(config["base_path"], trust_remote_code=False)
    train, validation, panel, stats = prepare_reranker_data(
        data, tokenizer, config["instruction"], config["max_length"], config["pair_limit"]
    )
    for name, rows in (("train", train), ("validation", validation)):
        path = run / f"{name}.jsonl"
        write_jsonl(path, rows)
        config[f"{name}_data"] = str(path)
    config["data_sha256"] = {
        key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
        for key in ("train_data", "validation_data")
    }
    if recipe and recipe["data_sha256"] != config["data_sha256"]:
        raise ValueError("Resume data differs; use the original dataset and pair limit.")
    set_reranker_environment()
    atomic_json(
        run / "prompt_check.json",
        verify_swift_prompt(
            config["base_path"], train[0], config["instruction"], config["max_length"]
        ),
    )
    config["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "ms-swift", "transformers", "peft", "bitsandbytes", "datasets")
    }
    atomic_json(run / "config.json", config)
    atomic_json(run / "data_statistics.json", stats)
    atomic_json(run / "evaluation_panel.json", panel)


def train_reranker_worker(config):
    set_reranker_environment()
    from swift.llm import sft_main
    from swift.plugin import extra_callbacks

    if time.time() >= config["train_deadline"]:
        raise RuntimeError("Training deadline expired before loading model.")
    extra_callbacks.append(
        deadline_callback(config["train_deadline"], Path(config["run"]) / "progress.json")
    )
    result = sft_main(reranker_arguments(config))
    atomic_json(Path(config["run"]) / "train_result.json", result)


def export_reranker_adapter(config, status):
    run = Path(config["run"])
    integrity_path = run / "training_integrity.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    if not integrity.get("adapter_updated"):
        raise RuntimeError("No verified adapter weight update; no model is exported.")
    checkpoints = [
        item
        for item in complete_checkpoints(config["output_dir"])
        if item[0] <= integrity.get("optimizer_steps", 0)
    ]
    if not checkpoints:
        raise RuntimeError("No complete resumable checkpoint; inspect train.log.")
    _, latest, state = checkpoints[-1]
    best = Path(state.get("best_model_checkpoint") or "").name
    step, selected, _ = next(
        (item for item in checkpoints if item[1].name == best), checkpoints[-1]
    )
    output = run / "qwen3-reranker-adapter"
    output.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(selected / name, output / name)
    atomic_json(
        output / "research_model.json",
        {
            "format": "swift_qwen3_reranker_adapter_v1",
            "model_family": "qwen3_reranker",
            "task_type": "generative_reranker",
            "loss_type": "generative_reranker",
            "loss": "BCEWithLogitsLoss(yes_logit - no_logit, binary_label)",
            "positive_token": "yes",
            "negative_token": "no",
            "fine_tuned": True,
            "optimizer_steps": step,
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "instruction": config["instruction"],
            "max_length": config["max_length"],
            "recipe": RECIPE,
            "selection": "best_eval_loss" if selected.name == best else "last_unvalidated",
            "quantization": "bnb NF4 double quantization, FP16 compute",
            "process_status": status,
            "packages": config.get("packages", {}),
            "data_sha256": config["data_sha256"],
            "negative_policy": "same-paper BM25 silver, all known gold excluded",
        },
    )
    if not (run / "resume").exists():
        shutil.copytree(latest, run / "resume")
    atomic_json(
        run / "resume_recipe.json",
        {
            **{
                key: config[key]
                for key in (
                    "base_model",
                    "base_revision",
                    "instruction",
                    "max_length",
                    "learning_rate",
                    "data_sha256",
                    "recipe",
                )
            },
            "dataset_revision": config.get("dataset_revision", ""),
        },
    )
    return output


def ranking_metrics(panel, scores=None):
    ndcg, mrr, recall, candidate_recall = [], [], [], []
    for i, item in enumerate(panel):
        gold = set(item["relevant_ids"])
        candidates = item["candidates"]
        order = list(range(len(candidates)))
        if scores is not None:
            if len(scores[i]) != len(candidates) or not all(math.isfinite(x) for x in scores[i]):
                raise ValueError("Missing or non-finite evaluation scores.")
            order.sort(key=lambda j: (-scores[i][j], j))
        relevance = [int(candidates[j]["id"] in gold) for j in order[:10]]
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(relevance))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(gold))))
        ndcg.append(dcg / ideal)
        mrr.append(next((1 / (rank + 1) for rank, hit in enumerate(relevance) if hit), 0))
        recall.append(sum(relevance) / len(gold))
        candidate_recall.append(len({c["id"] for c in candidates} & gold) / len(gold))
    if not panel:
        raise ValueError("Cannot evaluate an empty panel.")
    return {
        "ndcg@10": sum(ndcg) / len(panel),
        "mrr@10": sum(mrr) / len(panel),
        "recall@10": sum(recall) / len(panel),
        "candidate_recall@20": sum(candidate_recall) / len(panel),
    }


def evaluate_reranker_worker(config):
    run = Path(config["run"])
    panel = json.loads((run / "evaluation_panel.json").read_text(encoding="utf-8"))
    scorer = QwenRerankerScorer(
        config["base_path"],
        run / "qwen3-reranker-adapter",
        instruction=config["instruction"],
        max_length=config["max_length"],
    )

    def measure():
        scores = []
        for item in panel:
            values = []
            for document in item["candidates"]:
                if time.time() >= config["eval_deadline"]:
                    raise TimeoutError("Evaluation reserve reached; no partial metric is reported.")
                values.extend(scorer.score([(item["question"], document["text"])]))
            scores.append(values)
        return ranking_metrics(panel, scores)

    with scorer.model.disable_adapter():
        before = measure()
    after = measure()
    atomic_json(
        run / "reranking_validation.json",
        {
            "bm25": ranking_metrics(panel),
            "base": before,
            "fine_tuned": after,
            "queries": len(panel),
            "scope": "fixed validation subset with frozen BM25 candidates",
            "gold_injection": False,
            "test_set_used": False,
            "quantization": "same NF4 base for both evaluations",
        },
    )
