def patch_environment():
    """Patches missing triton.ops, removes invalid BNB_CUDA_VERSION overrides, and handles CPU/GPU fallbacks for Kaggle and Colab runtimes."""
    import os
    import sys
    import types
    import importlib.machinery

    # 1. Clean BNB_CUDA_VERSION if it causes missing libcudart.so
    if "BNB_CUDA_VERSION" in os.environ:
        del os.environ["BNB_CUDA_VERSION"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 2. Patch Triton ops mock for Triton 3+ compatibility
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

    # 3. Bulletproof fallback for bitsandbytes if native CUDA libs fail or not present
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


    # 4. Patch importlib.metadata.version fallback for bitsandbytes
    try:
        import importlib.metadata
        _orig_meta_version = importlib.metadata.version
        def _safe_meta_version(distribution_name):
            try:
                return _orig_meta_version(distribution_name)
            except Exception:
                if distribution_name == "bitsandbytes":
                    return "0.47.0"
                raise
        importlib.metadata.version = _safe_meta_version
    except Exception:
        pass

    # 5. Patch missing MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES in transformers for TRL DPO compatibility
    try:
        import transformers.models.auto.modeling_auto as mma
        if not hasattr(mma, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
            mma.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
    except Exception:
        pass

    # 6. Mock optional llm_blender for TRL PairwiseJudge compatibility
    if "llm_blender" not in sys.modules:
        try:
            import llm_blender
        except Exception:
            m_blender = types.ModuleType("llm_blender")
            m_blender.__spec__ = importlib.machinery.ModuleSpec("llm_blender", None)
            m_blender.Blender = type("Blender", (), {})
            sys.modules["llm_blender"] = m_blender

patch_environment()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent3TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 3 Generator & Critic on Kaggle T4x2")
    parser.add_argument("--role", choices=["generator", "critic", "both"], default="both")
    parser.add_argument("--epochs-sft", type=int, default=3)
    parser.add_argument("--epochs-dpo", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--max-length", type=int, default=1536, help="Max sequence length")
    
    parser.add_argument("--weights-dir", "--output-dir", dest="weights_dir", type=str, default=None)
    parser.add_argument("--dataset-dir", "--data", dest="dataset_dir", type=str, default=None)
    args = parser.parse_args()

    from src.trainer import train_generator_sft_and_dpo, train_critic_sft
    from src.evaluator import evaluate_generator_and_critic

    kwargs = {"gen_epochs_sft": args.epochs_sft, "gen_epochs_dpo": args.epochs_dpo, "critic_epochs": args.epochs_sft, "per_device_batch_size": args.batch_size, "max_seq_length": args.max_length, "critic_max_seq_length": args.max_length}
    if args.dataset_dir: kwargs["dataset_dir"] = Path(args.dataset_dir)
    if args.weights_dir: kwargs["weights_dir"] = Path(args.weights_dir)
    cfg = Agent3TrainingConfig(**kwargs)
    if args.role in ("generator", "both"):
        train_generator_sft_and_dpo(cfg)
    if args.role in ("critic", "both"):
        train_critic_sft(cfg)
    evaluate_generator_and_critic(cfg.weights_dir)


if __name__ == "__main__":
    main()
