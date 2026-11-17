import json
from pathlib import Path
# -*- coding: utf-8 -*-
"""
SFT Trainer & FP16 Compact Weight Exporter for Agent 1 (Retrieval Router).
Ensures `adapter_model.safetensors` is cast to `torch.float16` (< 80MB) for direct GitHub storage.
"""

import torch
import inspect
import dataclasses
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig
from .config import Agent1TrainingConfig
from .data_loader import load_router_dataset
from .model_builder import build_router_model_and_tokenizer


def safe_build_config(config_cls, **kwargs):
    """Safely instantiates TrainingArguments, SFTConfig, or DPOConfig across different transformers/trl versions."""
    try:
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    except TypeError:
        valid_fields = set(inspect.signature(config_cls.__init__).parameters.keys())

    if "max_length" in valid_fields and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    elif "max_seq_length" in valid_fields and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")

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


def export_fp16_safetensors(output_dir):
    """Casts saved adapter weights to FP16 so file size stays < 80MB (GitHub 100MB limit compliant)."""
    sf_path = output_dir / "adapter_model.safetensors"
    output_dir = Path(output_dir)
    bin_path = output_dir / "adapter_model.bin"
    if sf_path.exists() and bin_path.exists():
        bin_path.unlink() # purge stale .bin
    if sf_path.exists():
        tensors = load_file(str(sf_path))
        fp16_tensors = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in tensors.items()}
        save_file(fp16_tensors, str(sf_path))
        size_mb = sf_path.stat().st_size / (1024 * 1024)
        mb_size = sf_path.stat().st_size / (1024 * 1024)
        if mb_size > 99.0:
            print("⚠️ [WARNING] Adapter file exceeds GitHub 100MB limit. Use Git LFS or reduce LoRA rank.")
        print(f"📦 [Agent 1 Exporter] Exported FP16 adapter: {sf_path} ({size_mb:.2f} MB)")


def run_router_sft_training(cfg: Agent1TrainingConfig):
    """Executes full SFT training loop on Kaggle Dual T4 and exports FP16 adapter weights."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = build_router_model_and_tokenizer(cfg)
    train_ds, val_ds = load_router_dataset(cfg.dataset_path, tokenizer, cfg.val_split_ratio, cfg.seed)

    sft_args = safe_build_config(
        SFTConfig,
        output_dir=str(cfg.output_dir / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=cfg.weight_decay,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=None,
        **({"processing_class": tokenizer} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else {"tokenizer": tokenizer}),
        args=sft_args,
    )
    print("🚀 [Agent 1 Trainer] Starting SFT training on Qwen2.5-7B with DoRA + rsLoRA...")
    trainer.train()

    eval_res = trainer.evaluate()
    print(f"🏆 Best Evaluation Loss: {eval_res.get('eval_loss')}")
    final_adapter_dir = cfg.output_dir
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    export_fp16_safetensors(final_adapter_dir)
    def _json_default(obj):
        if hasattr(obj, "item"): return obj.item()
        if hasattr(obj, "__float__"): return float(obj)
        return str(obj)
    (final_adapter_dir / "eval_manifest.json").write_text(json.dumps(eval_res, indent=2, default=_json_default), encoding="utf-8")
    print(f"✅ [Agent 1 Trainer] Training completed and weights saved to {final_adapter_dir}")
    return trainer