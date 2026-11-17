import json
import gc
# -*- coding: utf-8 -*-
"""
Two-Stage SFT + RPO/DPO Trainer for Agent 5 (Academic Report Writer) on Kaggle Dual T4.
Exports compact FP16 `adapter_model.safetensors` (< 80MB) for direct GitHub storage and Multi-LoRA serving.
"""

import torch
import inspect
import dataclasses
from pathlib import Path
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from .config import Agent5TrainingConfig
from .data_loader import load_writer_sft_dataset, load_writer_dpo_dataset
from .model_builder import build_writer_model_and_tokenizer


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


def export_fp16_adapter(out_dir: Path):
    out_dir = Path(out_dir)
    sf_path = out_dir / "adapter_model.safetensors"
    bin_path = out_dir / "adapter_model.bin"
    if sf_path.exists() and bin_path.exists():
        bin_path.unlink() # purge stale .bin
    if sf_path.exists():
        tensors = load_file(str(sf_path))
        fp16_tensors = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in tensors.items()}
        save_file(fp16_tensors, str(sf_path))
        mb = sf_path.stat().st_size / (1024 * 1024)
        mb_size = sf_path.stat().st_size / (1024 * 1024)
        if mb_size > 99.0:
            print("⚠️ [WARNING] Adapter file exceeds GitHub 100MB limit. Use Git LFS or reduce LoRA rank.")
        print(f"📦 [Agent 5 Exporter] Saved FP16 Writer adapter: {sf_path} ({mb:.2f} MB)")


def _sanitize_model_dtypes(model):
    """Force all bf16 parameters to fp32 to prevent GradScaler unscale_ NotImplementedError on T4."""
    converted = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            converted += 1
    if converted > 0:
        print(f"🔧 [Dtype Sanitizer] Converted {converted} BFloat16 params to Float32 for GradScaler compatibility.")


def _detect_mixed_precision():
    """Detect the best mixed precision mode for the current GPU, avoiding GradScaler bf16 bugs."""
    if not torch.cuda.is_available():
        return False, False
    # T4 (compute capability 7.5) supports fp16 natively but bf16 only via emulation
    # Use fp16 by default; if params are bf16, _sanitize_model_dtypes will fix them
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8:
        # Ampere+ (A100, L4, etc.): bf16 is native and avoids GradScaler entirely
        print(f"⚡ [Mixed Precision] GPU compute capability {cap[0]}.{cap[1]} >= 8.0 -> using bf16 (no GradScaler)")
        return False, True  # fp16=False, bf16=True
    else:
        # Turing (T4) and older: use fp16 with GradScaler + dtype sanitization
        print(f"⚡ [Mixed Precision] GPU compute capability {cap[0]}.{cap[1]} < 8.0 -> using fp16 with GradScaler")
        return True, False  # fp16=True, bf16=False


def run_writer_sft_and_rpo_training(cfg: Agent5TrainingConfig):
    """Runs Stage 1 SFT followed by Stage 2 RPO/DPO Preference Alignment for Academic Report Synthesis."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = build_writer_model_and_tokenizer(cfg)

    # Sanitize bf16 params BEFORE creating trainers to prevent GradScaler crash
    _sanitize_model_dtypes(model)

    train_sft, val_sft = load_writer_sft_dataset(cfg.dataset_dir / "writer_sft.jsonl", tokenizer, seed=cfg.seed)

    use_fp16, use_bf16 = _detect_mixed_precision()

    sft_args = safe_build_config(
        SFTConfig,
        output_dir=str(cfg.output_dir / "sft_checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr_sft,
        num_train_epochs=cfg.epochs_sft,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        prediction_loss_only=True,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
    )
    sft_trainer = SFTTrainer(
        model=model, train_dataset=train_sft, eval_dataset=val_sft, peft_config=None, **({"processing_class": tokenizer} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else {"tokenizer": tokenizer}), args=sft_args
    )
    sft_ckpt_dir = cfg.output_dir / "sft_checkpoints"
    has_ckpt = any(sft_ckpt_dir.glob("checkpoint-*")) if sft_ckpt_dir.exists() else False
    print(f"🚀 [Agent 5 Writer] Starting Stage 1 SFT (Tự động tiếp tục từ checkpoint: {has_ckpt})...")

    # Try training with detected precision; fall back to fp32 if GradScaler still fails
    try:
        sft_trainer.train(resume_from_checkpoint=has_ckpt)
    except (NotImplementedError, RuntimeError) as e:
        if "BFloat16" in str(e) or "unscale" in str(e):
            print(f"⚠️ [Fallback] GradScaler error detected: {e}")
            print("🔄 [Fallback] Restarting SFT with fp32 (no mixed precision)...")
            sft_args_fp32 = safe_build_config(
                SFTConfig,
                output_dir=str(cfg.output_dir / "sft_checkpoints"),
                per_device_train_batch_size=cfg.per_device_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.lr_sft,
                num_train_epochs=cfg.epochs_sft,
                max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
                fp16=False,
                bf16=False,
                logging_steps=10,
                per_device_eval_batch_size=1,
                eval_strategy="no",
                prediction_loss_only=True,
                max_length=cfg.max_seq_length,
                dataset_text_field="text",
                optim="paged_adamw_8bit",
                lr_scheduler_type="cosine",
                warmup_steps=10,
                neftune_noise_alpha=cfg.neftune_noise_alpha,
                gradient_checkpointing=True,
                report_to="none",
            )
            sft_trainer = SFTTrainer(
                model=model, train_dataset=train_sft, eval_dataset=val_sft, peft_config=None, **({"processing_class": tokenizer} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else {"tokenizer": tokenizer}), args=sft_args_fp32
            )
            sft_trainer.train()
        else:
            raise

    final_model = sft_trainer.model

    dpo_file = cfg.dataset_dir / "writer_dpo.jsonl"
    dpo_successful = False
    dpo_trainer_inst = None
    if dpo_file.exists() and cfg.epochs_dpo > 0:
        try:
            print("🎯 [Agent 5 Writer] Starting Stage 2 RPO/DPO Citation Alignment...")
            train_dpo, val_dpo = load_writer_dpo_dataset(dpo_file, seed=cfg.seed)
            if len(train_dpo) == 0:
                raise ValueError("DPO dataset has 0 valid samples, skipping Stage 2.")
            dpo_args = safe_build_config(
                DPOConfig,
                output_dir=str(cfg.output_dir / "dpo_checkpoints"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=cfg.lr_dpo,
                num_train_epochs=cfg.epochs_dpo,
                beta=cfg.dpo_beta,
                rpo_alpha=0.1,  # RPO alignment
                fp16=use_fp16,
                bf16=use_bf16,
                logging_steps=10,
                max_length=1536,
                max_prompt_length=768,
                optim="paged_adamw_8bit",
                lr_scheduler_type="cosine",
                warmup_steps=10,
                gradient_checkpointing=True,
                report_to="none",
            )
            dpo_trainer_inst = DPOTrainer(
                model=sft_trainer.model, ref_model=None, args=dpo_args, train_dataset=train_dpo, eval_dataset=val_dpo, **({"processing_class": tokenizer} if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else {"tokenizer": tokenizer})
            )
            dpo_trainer_inst.train()
            final_model = dpo_trainer_inst.model
            dpo_successful = True
        except Exception as e:
            print(f"ℹ️ [Agent 5 Writer] DPO notice ({e}). Lưu trọng số SFT hoàn chỉnh.")
    else:
        print("⏩ [Agent 5 Writer] Bỏ qua Stage 2 DPO theo yêu cầu (epochs_dpo = 0). Tiến hành đóng gói mô hình SFT.")

    active_trainer = dpo_trainer_inst if dpo_successful else sft_trainer
    active_trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    export_fp16_adapter(cfg.output_dir)
    try:
        eval_res = active_trainer.evaluate()
    except Exception:
        eval_res = {"status": "trained"}
    def _json_default(obj):
        if hasattr(obj, "item"): return obj.item()
        if hasattr(obj, "__float__"): return float(obj)
        return str(obj)
    (cfg.output_dir / "eval_manifest.json").write_text(json.dumps(eval_res, indent=2, default=_json_default), encoding="utf-8")
    del sft_trainer
    if dpo_trainer_inst is not None:
        del dpo_trainer_inst
    if "final_model" in locals():
        del final_model
    if "model" in locals():
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ [Agent 5 Writer] Training finished, eval manifest and FP16 adapter saved to {cfg.output_dir}")
