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


import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent2TrainingConfig


def sanitize_torchvision():
    """Tự động vô hiệu hóa cờ torchvision trong transformers để nạp cực nhanh và tránh xung đột C++ operator."""
    try:
        import transformers.utils.import_utils as tu
        tu._torchvision_available = False
        tu.is_torchvision_available = lambda: False
    except Exception:
        pass


def main():
    sanitize_torchvision()
    print("🚀 [Agent 2 Reader] Đang khởi tạo mô hình LayoutLMv3...", flush=True)
    parser = argparse.ArgumentParser(description="Fine-tune Agent 2 LayoutLMv3 Reader on Kaggle T4x2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from src.trainer import run_reader_training
    from src.evaluator import evaluate_layout_parser

    kwargs = {"epochs": args.epochs, "learning_rate": args.lr, "per_device_train_batch_size": args.batch_size}
    if args.data: kwargs["dataset_path"] = Path(args.data)
    if args.output_dir: kwargs["output_dir"] = Path(args.output_dir)
    cfg = Agent2TrainingConfig(**kwargs)
    model, tokenizer = run_reader_training(cfg)
    evaluate_layout_parser(model, tokenizer)


if __name__ == "__main__":
    main()
