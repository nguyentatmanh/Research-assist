import gc
# -*- coding: utf-8 -*-
"""
SOTA Weighted Focal + R-Drop Cross-Encoder Trainer for Agent 4 (Fact-Checker NLI Verifier) on Kaggle Dual T4.
Combines:
1. Class-Weighted Focal Loss (gamma=2.0, CONTRADICT weight = 1.35x)
2. R-Drop Regularization (Bidirectional KL-divergence consistency between two forward passes)
3. Post-training Temperature Scaling Calibration (ECE minimization)
"""

import json
import torch
import inspect
import dataclasses
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .config import Agent4TrainingConfig
from .data_loader import load_nli_dataset
from .model_builder import build_nli_cross_encoder
from .evaluator import compute_nli_metrics, calibrate_temperature_ece


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


class WeightedCrossEncoderTrainer(Trainer):
    """SOTA Trainer with Class-Weighted Focal Loss (gamma=2.0) + R-Drop KL Consistency Regularization."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs1 = model(**inputs)
        logits1 = outputs1.get("logits") if hasattr(outputs1, "get") else outputs1.logits

        if labels is None:
            return (outputs1.loss, outputs1) if return_outputs else outputs1.loss

        num_labels = getattr(getattr(model, "config", None), "num_labels", 3)
        weights = torch.tensor([1.0, 1.35, 1.15], device=logits1.device, dtype=logits1.dtype) if num_labels == 3 else None
        active_mask = (labels.view(-1) != -100)
        ce1 = F.cross_entropy(logits1.view(-1, num_labels), labels.view(-1), weight=weights, reduction="none", ignore_index=-100)
        pt1 = torch.exp(-ce1)
        focal_terms1 = (1.0 - pt1) ** 2.0 * ce1
        focal1 = focal_terms1[active_mask].sum() / active_mask.sum().clamp(min=1)

        if model.training:
            # R-Drop second forward pass for KL consistency regularization
            inputs_clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs2 = model(**inputs_clone)
            logits2 = outputs2.get("logits") if hasattr(outputs2, "get") else outputs2.logits
            ce2 = F.cross_entropy(logits2.view(-1, num_labels), labels.view(-1), weight=weights, reduction="none", ignore_index=-100)
            pt2 = torch.exp(-ce2)
            focal_terms2 = (1.0 - pt2) ** 2.0 * ce2
            focal2 = focal_terms2[active_mask].sum() / active_mask.sum().clamp(min=1)

            p1 = F.log_softmax(logits1, dim=-1)
            p2 = F.log_softmax(logits2, dim=-1)
            # Numerically stable KL with log_target=True avoids FP16 underflow
            kl_loss = 0.5 * (F.kl_div(p1, p2, log_target=True, reduction="batchmean") + F.kl_div(p2, p1, log_target=True, reduction="batchmean"))
            loss = 0.5 * (focal1 + focal2) + 0.3 * kl_loss
        else:
            loss = focal1

        return (loss, outputs1) if return_outputs else loss


def run_factchecker_training(cfg: Agent4TrainingConfig):
    """Executes full Cross-Encoder NLI training with R-Drop + Focal Loss on Kaggle Dual T4."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_nli_cross_encoder(cfg)
    train_ds, val_ds = load_nli_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

    training_args = safe_build_config(
        TrainingArguments,
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=20,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = WeightedCrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **(dict(processing_class=tokenizer) if "processing_class" in inspect.signature(Trainer.__init__).parameters else dict(tokenizer=tokenizer)),
        compute_metrics=compute_nli_metrics,
    )
    print("🚀 [Agent 4 Trainer] Starting R-Drop + Focal Loss NLI training...")
    trainer.train()
    metrics = trainer.evaluate()

    opt_temp = 1.0
    try:
        preds_out = trainer.predict(val_ds)
        preds = preds_out.predictions[0] if isinstance(preds_out.predictions, tuple) else preds_out.predictions
        opt_temp = calibrate_temperature_ece(preds, preds_out.label_ids)
        del preds
        del preds_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"ℹ️ [Agent 4 Trainer] Temperature calibration notice: {e}")

    metrics["calibrated_temperature"] = opt_temp
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__float__"):
            return float(obj)
        return str(obj)
    (cfg.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    print(f"✅ [Agent 4 Trainer] Saved SOTA R-Drop + Focal-Loss NLI Verifier (T={opt_temp}) to {cfg.output_dir}")
    return trainer, metrics
