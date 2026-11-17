# -*- coding: utf-8 -*-
"""
Model Builder for Agent 5 (Academic Report Writer).
Loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 (~4.8GB) on `cuda:0` for zero-latency, single-device execution.
Ensures all trainable adapter parameters are strictly `torch.float32` master weights for PyTorch AMP GradScaler.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import Agent5TrainingConfig


def get_dual_t4_device_map(max_mem: str = "14.0GiB"):
    """Allocates model on a single GPU to avoid TRL multi-device per_token_loss bug."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        target = torch.cuda.current_device()
        print(f"⚡ [ModelBuilder] Using single GPU cuda:{target} (available: {gpu_count}) to avoid TRL multi-device loss bug...")
        return {"": target}, None
    return "auto", None


def build_writer_model_and_tokenizer(cfg: Agent5TrainingConfig):
    """Initializes Qwen2.5-7B-Instruct with SDPA Attention and attaches DoRA + rsLoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    dev_map, max_mem = get_dual_t4_device_map(cfg.max_memory_per_gpu)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_config,
        device_map=dev_map,
        max_memory=max_mem,
        torch_dtype=torch.float16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    )
    model.config.torch_dtype = torch.float16
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_dora=cfg.use_dora,
        use_rslora=(cfg.use_rslora and not cfg.use_dora),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Kích hoạt bắt buộc input require grad cho gradient checkpointing PEFT QLoRA
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    try:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    except Exception:
        pass
    model.config.use_cache = False

    # Chuẩn AMP của PyTorch trên Tesla T4: Trọng số LoRA trainable phải ở float32 để GradScaler unscale:
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model, tokenizer, peft_config
