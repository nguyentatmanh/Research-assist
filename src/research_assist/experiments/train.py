"""Fine-tune an E5 retriever with MNRL; validation chooses checkpoint, test stays untouched."""

import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research_assist.experiments.data import file_hash, load_split, validated_pairs


@dataclass
class TrainConfig:
    data: str
    output: str
    model: str = "intfloat/multilingual-e5-small"
    revision: str = ""
    epochs: float = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    device: str = "cuda"
    seed: int = 42
    synthetic: str | None = None


def train(config: TrainConfig) -> dict[str, Any]:
    # Training dependencies are optional and never imported by the API.
    import sentence_transformers.losses as losses
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers
    from transformers import set_seed

    if config.batch_size < 2 or config.epochs <= 0 or config.max_seq_length < 32:
        raise ValueError("Require batch_size >= 2, epochs > 0 and max_seq_length >= 32")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose --device cpu or enable Kaggle GPU")
    set_seed(config.seed)
    options: dict[str, Any] = {"device": config.device, "trust_remote_code": False}
    local_base = Path(config.model)
    revision = config.revision
    if not local_base.is_dir() and not revision:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(config.model).sha or ""
    if revision:
        options["revision"] = revision
    model = SentenceTransformer(config.model, **options)
    model.max_seq_length = config.max_seq_length
    data = Path(config.data)
    pairs = validated_pairs(data, Path(config.synthetic) if config.synthetic else None)
    original = len(pairs)
    # Don't train on evidence whose answer tail the tokenizer would discard.
    pairs = [
        p
        for p in pairs
        if len(model.tokenizer.encode("query: " + p["question"])) <= config.max_seq_length
        and len(model.tokenizer.encode("passage: " + p["positive"])) <= config.max_seq_length
    ]
    if len(pairs) < 2:
        raise ValueError("Too few non-truncated training pairs")
    dataset = Dataset.from_dict(
        {
            "anchor": ["query: " + p["question"] for p in pairs],
            "positive": ["passage: " + p["positive"] for p in pairs],
        }
    )
    corpus, queries = load_split(data, "validation")
    evaluator = InformationRetrievalEvaluator(
        queries={q["id"]: "query: " + q["question"] for q in queries},
        corpus={c["id"]: "passage: " + c["text"] for c in corpus},
        relevant_docs={q["id"]: set(q["relevant_ids"]) for q in queries},
        name="validation",
        show_progress_bar=False,
        batch_size=config.batch_size,
        write_csv=True,
    )
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; choose a new experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    baseline = evaluator(model, output_path=str(output))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        fp16=config.device == "cuda",
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=evaluator.primary_metric,
        greater_is_better=True,
        logging_steps=10,
        seed=config.seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=losses.MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
    )
    trained = trainer.train()
    model = trainer.model
    after = evaluator(model, output_path=str(output))
    model.save_pretrained(str(output))
    manifest = {
        "format_version": 2,
        "model_family": "e5",
        "fine_tuned": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "base_model": config.model,
        "revision": revision,
        "base_manifest": json.loads((local_base / "research_model.json").read_text())
        if (local_base / "research_model.json").is_file()
        else None,
        "max_seq_length": config.max_seq_length,
        "config": asdict(config),
        "pairs_used": len(pairs),
        "pairs_excluded_truncation": original - len(pairs),
        "corpus_sha256": file_hash(data / "corpus.jsonl"),
        "pairs_sha256": file_hash(data / "train_pairs.jsonl"),
        "synthetic_sha256": file_hash(Path(config.synthetic)) if config.synthetic else None,
        "validation_before": baseline,
        "validation_after": after,
        "training_metrics": trained.metrics,
        "versions": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ("torch", "sentence-transformers", "datasets", "transformers", "accelerate")
        },
    }
    (output / "research_model.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=TrainConfig.model)
    parser.add_argument("--revision", default="")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", default=None)
    config = TrainConfig(**vars(parser.parse_args()))
    manifest = train(config)
    print(
        json.dumps(
            {
                "output": config.output,
                "pairs_used": manifest["pairs_used"],
                "validation_before": manifest["validation_before"],
                "validation_after": manifest["validation_after"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
