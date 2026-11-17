def patch_environment():
    """Patches missing triton.ops, language attributes, and handles CPU/GPU fallbacks for Kaggle and Colab."""
    import os
    import sys
    import types
    import importlib.machinery

    if "BNB_CUDA_VERSION" in os.environ:
        del os.environ["BNB_CUDA_VERSION"]

    # 1. Patch Triton mock safely
    m_triton = sys.modules.get("triton")
    if m_triton is None:
        try:
            import triton
            m_triton = triton
        except Exception:
            m_triton = types.ModuleType("triton")
            m_triton.__spec__ = importlib.machinery.ModuleSpec("triton", None)
            m_triton.__path__ = []
            sys.modules["triton"] = m_triton

    # Ensure triton.language exists to satisfy transformers.modeling_utils import checks
    if not hasattr(m_triton, "language"):
        m_lang = types.ModuleType("triton.language")
        m_lang.__spec__ = importlib.machinery.ModuleSpec("triton.language", None)
        m_lang.constexpr = lambda x: x
        setattr(m_triton, "language", m_lang)
        sys.modules["triton.language"] = m_lang

    if not hasattr(m_triton, "cdiv"):
        setattr(m_triton, "cdiv", lambda a, b: (a + b - 1) // b)
    if not hasattr(m_triton, "jit"):
        setattr(m_triton, "jit", lambda *a, **k: (lambda f: f))

    m_ops = types.ModuleType("triton.ops")
    m_ops.__spec__ = importlib.machinery.ModuleSpec("triton.ops", None)
    m_ops.__path__ = []
    m_perf = types.ModuleType("triton.ops.matmul_perf_model")
    m_perf.__spec__ = importlib.machinery.ModuleSpec("triton.ops.matmul_perf_model", None)
    m_perf.early_config_prune = lambda *a, **k: None
    m_perf.estimate_matmul_time = lambda *a, **k: 0
    m_ops.matmul_perf_model = m_perf
    setattr(m_triton, "ops", m_ops)
    sys.modules["triton.ops"] = m_ops
    sys.modules["triton.ops.matmul_perf_model"] = m_perf

    # 2. Bulletproof fallback for bitsandbytes
    try:
        import bitsandbytes as bnb
    except Exception:
        import torch
        m_bnb = types.ModuleType("bitsandbytes")
        m_bnb.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)
        m_bnb_nn = types.ModuleType("bitsandbytes.nn")
        m_bnb_nn.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.nn", None)

        class DummyBnbLinear(torch.nn.Linear):
            pass

        m_bnb_nn.Linear8bitLt = DummyBnbLinear
        m_bnb_nn.Linear4bit = DummyBnbLinear
        m_bnb_nn.modules = types.ModuleType("bitsandbytes.nn.modules")
        m_bnb_nn.modules.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.nn.modules", None)
        m_bnb.nn = m_bnb_nn
        m_bnb.__version__ = "0.47.0"
        sys.modules["bitsandbytes"] = m_bnb
        sys.modules["bitsandbytes.nn"] = m_bnb_nn
        sys.modules["bitsandbytes.nn.modules"] = m_bnb_nn.modules

patch_environment()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Chạy SFT Router với hỗ trợ phân bổ tự động Dual GPU T4x2 hoặc Single T4:

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent1TrainingConfig


def sanitize_torchvision():
    """Tự động vô hiệu hóa cờ torchvision trong transformers nếu torchvision bị lỗi C++ operator."""
    try:
        import torchvision
        _ = getattr(torchvision.ops, "nms", None)
    except Exception:
        try:
            import transformers.utils.import_utils as tu
            tu._torchvision_available = False
            tu.is_torchvision_available = lambda: False
        except Exception:
            pass


def main():
    sanitize_torchvision()
    parser = argparse.ArgumentParser(description="Fine-tune Agent 1 Retrieval Router on Kaggle T4x2 or Colab T4")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA/DoRA rank r")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--data", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    try:
        import trl
    except ImportError:
        import subprocess
        print("⏳ Đang tự động cài đặt thư viện trl...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl==0.12.1"])

    try:
        import bitsandbytes
    except Exception:
        import subprocess
        print("⏳ Đang tự động cài đặt bitsandbytes...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])

    from src.trainer import run_router_sft_training
    from src.evaluator import evaluate_router_adapter

    kwargs = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "per_device_train_batch_size": args.batch_size,
        "max_seq_length": args.max_length,
    }
    if args.data: kwargs["dataset_path"] = Path(args.data)
    if args.output_dir: kwargs["output_dir"] = Path(args.output_dir)
    cfg = Agent1TrainingConfig(**kwargs)
    if args.eval_only:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(cfg.output_dir))
        evaluate_router_adapter(model, tokenizer)
    else:
        trainer = run_router_sft_training(cfg)
        tok = getattr(trainer, "processing_class", getattr(trainer, "tokenizer", None))
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg.base_model_id)
        evaluate_router_adapter(trainer.model, tok)


if __name__ == "__main__":
    main()
