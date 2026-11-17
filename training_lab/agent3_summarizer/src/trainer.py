import gc
# -*- coding: utf-8 -*-
"""
Two-Stage SFT + DPO Trainer for Generator (`generator_qlora`) & SFT Trainer for Critic (`critic_qlora`).
Exports FP16 `adapter_model.safetensors` (< 80MB) directly to `weights/`.
"""

import torch
import inspect
import dataclasses
from pathlib import Path
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from .config import Agent3TrainingConfig
from .data_loader import load_sft_jsonl_dataset, load_dpo_jsonl_dataset
from .model_builder import build_summarizer_model_and_tokenizer


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


def cast_adapter_to_fp16(out_dir: Path):
    sf_path = out_dir / "adapter_model.safetensors"
    out_dir = Path(out_dir)
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
        print(f"📦 [Agent 3 Exporter] Saved FP16 adapter: {sf_path} ({mb:.2f} MB)")


def train_generator_sft_and_dpo(cfg: Agent3TrainingConfig):
    """Runs Stage 1 SFT followed by Stage 2 Direct Preference Optimization (DPO) for Generator."""
    out_dir = cfg.weights_dir / "generator_qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, peft_config = build_summarizer_model_and_tokenizer(
        cfg.base_model_id, cfg.gen_lora_r, cfg.gen_lora_alpha, cfg.lora_dropout, cfg.use_dora, cfg.max_memory_per_gpu
    )
    train_sft, val_sft = load_sft_jsonl_dataset(cfg.dataset_dir / "generator_sft.jsonl", tokenizer, seed=cfg.seed)

    sft_args = safe_build_config(
        SFTConfig,
        output_dir=str(out_dir / "sft_checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.gen_lr_sft,
        num_train_epochs=cfg.gen_epochs_sft,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=50,
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
    sft_ckpt_dir = out_dir / "sft_checkpoints"
    has_ckpt = any(sft_ckpt_dir.glob("checkpoint-*")) if sft_ckpt_dir.exists() else False
    print(f"🚀 [Agent 3 Generator] Starting Stage 1 SFT (Tự động tiếp tục từ checkpoint: {has_ckpt})...")
    sft_trainer.train(resume_from_checkpoint=has_ckpt)

    final_model = sft_trainer.model

    # Stage 2: DPO Alignment (bọc try-except bảo vệ)
    dpo_file = cfg.dataset_dir / "generator_dpo.jsonl"
    dpo_successful = False
    dpo_trainer_inst = None
    if dpo_file.exists():
        try:
            print("🎯 [Agent 3 Generator] Starting Stage 2 RPO/DPO Alignment...")
            train_dpo, val_dpo = load_dpo_jsonl_dataset(dpo_file, seed=cfg.seed)
            if len(train_dpo) == 0:
                raise ValueError("DPO dataset has 0 valid samples, skipping Stage 2.")
            dpo_args = safe_build_config(
                DPOConfig,
                output_dir=str(out_dir / "dpo_checkpoints"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=cfg.gen_lr_dpo,
                num_train_epochs=cfg.gen_epochs_dpo,
                beta=cfg.dpo_beta,
                rpo_alpha=0.1,  # RPO (Regularized Preference Optimization)
                fp16=torch.cuda.is_available(),
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
            print(f"ℹ️ [Agent 3 Generator] DPO notice ({e}). Lưu trọng số SFT hoàn chỉnh.")

    (dpo_trainer_inst if dpo_successful else sft_trainer).save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    cast_adapter_to_fp16(out_dir)
    del sft_trainer
    if "dpo_trainer" in locals():
        del dpo_trainer
    del final_model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_critic_sft(cfg: Agent3TrainingConfig):
    """Runs Hard-Negative Adversarial SFT for Critic (`critic_qlora`)."""
    out_dir = cfg.weights_dir / "critic_qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, peft_config = build_summarizer_model_and_tokenizer(
        cfg.base_model_id, cfg.critic_lora_r, cfg.critic_lora_alpha, cfg.lora_dropout, cfg.use_dora, cfg.max_memory_per_gpu
    )
    train_ds, val_ds = load_sft_jsonl_dataset(cfg.dataset_dir / "critic_sft.jsonl", tokenizer, seed=cfg.seed)

    sft_args = safe_build_config(
        SFTConfig,
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.critic_lr,
        num_train_epochs=cfg.critic_epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        prediction_loss_only=True,
        max_length=cfg.critic_max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=val_ds, peft_config=None, **({"processing_class": tokenizer} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else {"tokenizer": tokenizer}), args=sft_args
    )
    print("🚀 [Agent 3 Critic] Starting Adversarial SFT...")
    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    cast_adapter_to_fp16(out_dir)
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
