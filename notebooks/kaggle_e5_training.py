"""Notebook E5 training: cached negatives, cooperative deadlines and explicit selection status."""

import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from notebooks.kaggle_training_helpers import (
    NotebookDeadlineReached,
    evaluate_with_budget,
    make_time_budget_callback,
)
from research_assist.experiments.data import file_hash, load_split, validated_pairs


@dataclass
class TrainConfig:
    data: str
    output: str
    model: str = "intfloat/multilingual-e5-base"
    revision: str = ""
    epochs: float = 2
    batch_size: int = 32
    mini_batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    device: str = "cuda"
    seed: int = 42
    synthetic: str | None = None


def train(config, time_budget=None, checkpoint_minutes=15):
    import sentence_transformers.losses as losses
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SentenceEvaluator
    from sentence_transformers.training_args import BatchSamplers
    from transformers import set_seed

    if config.batch_size < 2 or config.mini_batch_size < 1 or config.epochs <= 0:
        raise ValueError("Require batch >= 2, mini batch >= 1 and epochs > 0")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; enable Kaggle GPU")
    set_seed(config.seed)
    options = {"device": config.device, "trust_remote_code": False}
    revision = config.revision
    if not Path(config.model).is_dir() and not revision:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(config.model).sha or ""
    if revision:
        options["revision"] = revision
    model = SentenceTransformer(config.model, **options)
    model.max_seq_length = config.max_seq_length
    data = Path(config.data)
    pairs = validated_pairs(data, Path(config.synthetic) if config.synthetic else None)
    original = len(pairs)
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
        batch_size=config.mini_batch_size,
        write_csv=True,
    )

    class TimedEvaluator(SentenceEvaluator):
        primary_metric = evaluator.primary_metric

        def __call__(self, current_model, *args, **kwargs):
            result = evaluate_with_budget(
                current_model, lambda: evaluator(current_model, *args, **kwargs), time_budget
            )
            self.primary_metric = evaluator.primary_metric
            return result

    timed = TimedEvaluator()
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; choose a new RUN_NAME")
    output.mkdir(parents=True, exist_ok=True)
    baseline, after, metrics, steps = None, None, {}, 0
    selection, stop_reason = "base_not_trained", "epochs_completed"
    trainer = None
    try:
        if time_budget and time_budget.training_expired():
            raise NotebookDeadlineReached("Training deadline already reached")
        baseline = timed(model, output_path=str(output))
        if time_budget and time_budget.training_expired():
            raise NotebookDeadlineReached("Training deadline reached during setup/validation")
        args = SentenceTransformerTrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.mini_batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=0.1,
            fp16=config.device == "cuda",
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=timed.primary_metric,
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
            loss=losses.CachedMultipleNegativesRankingLoss(
                model, mini_batch_size=config.mini_batch_size
            ),
            evaluator=timed,
            callbacks=[make_time_budget_callback(time_budget, checkpoint_minutes)]
            if time_budget
            else [],
        )
        trained = trainer.train()
        metrics = trained.metrics
    except NotebookDeadlineReached as exc:
        stop_reason = "time_budget"
        print(str(exc), flush=True)
        if trainer is not None:
            # Preserve current weights when evaluation was interrupted.
            # This is explicit recovery data, not an evaluated best model.
            trainer.save_model(str(output.parent / "recovery-e5"))
            trainer.save_state()
    if trainer is not None:
        steps = int(trainer.state.global_step)
        checkpoint = trainer.state.best_model_checkpoint
        if checkpoint:
            # Also works when a deadline interrupted Trainer before its normal best-model reload.
            model = SentenceTransformer(checkpoint, **{"device": config.device})
            selection = "best_validation_checkpoint"
        else:
            model = trainer.model
            selection = "last_unvalidated" if steps else "base_not_trained"
        if time_budget and time_budget.training_expired():
            stop_reason = "time_budget"
    try:
        if steps:
            after = timed(model, output_path=str(output))
    except NotebookDeadlineReached:
        print("Final validation skipped/incomplete; no partial metric is reported.", flush=True)
    if after is not None:
        selection = (
            "best_validation_checkpoint" if selection.startswith("best_") else "last_validated"
        )
    model.save_pretrained(str(output))
    manifest = {
        "format_version": 2,
        "model_family": "e5",
        "fine_tuned": steps > 0,
        "loss": "CachedMultipleNegativesRankingLoss",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "base_model": config.model,
        "revision": revision,
        "max_seq_length": config.max_seq_length,
        "config": asdict(config),
        "pairs_used": len(pairs),
        "pairs_excluded_truncation": original - len(pairs),
        "corpus_sha256": file_hash(data / "corpus.jsonl"),
        "pairs_sha256": file_hash(data / "train_pairs.jsonl"),
        "synthetic_sha256": file_hash(Path(config.synthetic)) if config.synthetic else None,
        "validation_before": baseline,
        "validation_after": after,
        "training_metrics": metrics,
        "optimizer_steps": steps,
        "selection": selection,
        "stop_reason": stop_reason,
        "selected_validation_metric": trainer.state.best_metric if trainer else None,
        "time_budget": time_budget.snapshot() if time_budget else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "datasets", "transformers", "accelerate")
        },
    }
    (output / "research_model.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest
