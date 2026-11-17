# -*- coding: utf-8 -*-
"""
SOTA Training Pipeline for Agent 2 (LayoutLMv3 Document Reader) on Kaggle Dual T4.
Includes Multi-Class Focal Loss (gamma=2.0) to boost rare scientific layout classes (`Formula`, `Table`, `Footnote`).
Evaluates and selects checkpoints via token-level Macro F1.
"""

import json
import torch
import inspect
import dataclasses
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .config import Agent2TrainingConfig, DOCLAYNET_LABELS
from .data_loader import load_doclaynet_dataset
from .model_builder import build_layoutlmv3_model
from .evaluator import compute_layout_metrics


def safe_build_config(config_cls, **kwargs):
    """Safely instantiates TrainingArguments across different transformers versions."""
    try:
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    except TypeError:
        valid_fields = set(inspect.signature(config_cls.__init__).parameters.keys())

    if "warmup_steps" in kwargs and "warmup_ratio" in kwargs:
        kwargs.pop("warmup_ratio")  # Prevent Transformers 'Both warmup_ratio and warmup_steps were set' ValueError
    elif "warmup_steps" in valid_fields and "warmup_ratio" in kwargs and "warmup_ratio" not in valid_fields:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" not in kwargs:
            kwargs["warmup_steps"] = max(1, int(ratio * 100))

    if "eval_strategy" in valid_fields and "evaluation_strategy" in kwargs and "evaluation_strategy" not in valid_fields:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in valid_fields and "eval_strategy" in kwargs and "eval_strategy" not in valid_fields:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return config_cls(**filtered)


class FocalLossLayoutTrainer(Trainer):
    """Custom Trainer using Multi-Class Focal Loss (gamma=2.0) for imbalanced document layout tokens."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
        if labels is not None:
            active_mask = (labels.view(-1) != -100)
            ce_loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), reduction="none", ignore_index=-100)
            pt = torch.exp(-ce_loss)
            focal_terms = (1.0 - pt) ** 2.0 * ce_loss
            focal_loss = focal_terms[active_mask].sum() / active_mask.sum().clamp(min=1)
            return (focal_loss, outputs) if return_outputs else focal_loss
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def run_reader_training(cfg: Agent2TrainingConfig):
    """Executes full Multi-Class Focal Loss training on LayoutLMv3 and exports weights + label_map.json."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_layoutlmv3_model(cfg)
    train_ds, val_ds = load_doclaynet_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

    training_args = safe_build_config(
        TrainingArguments,
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=(torch.cuda.is_available() and (torch.cuda.device_count() <= 1 or 'LOCAL_RANK' in os.environ)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_steps=10,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=20,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = FocalLossLayoutTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **(dict(processing_class=tokenizer) if "processing_class" in inspect.signature(Trainer.__init__).parameters else dict(tokenizer=tokenizer)),
        compute_metrics=compute_layout_metrics,
    )
    print("🚀 [Agent 2 Trainer] Starting Multi-Class Focal Loss training on LayoutLMv3 (Evaluating by Macro F1)...")
    trainer.train()

    eval_res = trainer.evaluate()
    print(f"🏆 Best LayoutLMv3 Validation Metrics: Macro F1 = {eval_res.get('eval_macro_f1')}")

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    label_map_path = cfg.output_dir / "label_map.json"
    label_map_path.write_text(json.dumps(DOCLAYNET_LABELS, indent=2), encoding="utf-8")
    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__float__"):
            return float(obj)
        return str(obj)
    (cfg.output_dir / "eval_manifest.json").write_text(json.dumps(eval_res, indent=2, default=_json_default), encoding="utf-8")
    print(f"✅ [Agent 2 Trainer] Model and label map saved to {cfg.output_dir}")
    return model, tokenizer
