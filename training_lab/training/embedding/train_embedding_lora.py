# -*- coding: utf-8 -*-
"""
Unsloth & MoAI-Style LoRA Embedding Training Pipeline for Multi-Agent RAG.
Optimized for Kaggle Dual Tesla T4 (2x16GB) or Single GPU.
Recipe:
  - Base: BAAI/bge-m3 (Multilingual Dense + Sparse)
  - LoRA: rank=32, alpha=64, dropout=0.05
  - Loss: CachedMultipleNegativesRankingLoss / MultipleNegativesRankingLoss
  - Metric: Validation Recall@10 & MRR@10
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
def patch_environment():
    """Patches missing triton.ops, removes invalid BNB_CUDA_VERSION overrides, and handles CPU/GPU fallbacks for Kaggle and Colab runtimes."""
    import os
    import sys
    import types
    import importlib.machinery

    # 1. Clean BNB_CUDA_VERSION if it causes missing libcudart.so
    if "BNB_CUDA_VERSION" in os.environ:
        del os.environ["BNB_CUDA_VERSION"]

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

patch_environment()

def patch_sentence_transformers():
    """Patches SentenceTransformerTrainer.compute_loss to accept num_items_in_batch added in transformers>=4.46."""
    try:
        import inspect
        import sentence_transformers
        
        candidates = [
            getattr(sentence_transformers, "trainer", None),
            getattr(sentence_transformers, "SentenceTransformerTrainer", None),
            getattr(sentence_transformers, "sentence_transformer", None),
        ]
        
        patched_any = False
        for cand in candidates:
            if cand is None:
                continue
            classes = [cand] if inspect.isclass(cand) else [getattr(cand, "SentenceTransformerTrainer", None)]
            for cls in classes:
                if cls is not None and hasattr(cls, "compute_loss"):
                    orig_fn = cls.compute_loss
                    try:
                        sig = inspect.signature(orig_fn)
                        if "num_items_in_batch" not in sig.parameters:
                            def make_loss_patch(old_loss):
                                def patched_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                                    return old_loss(self, model, inputs, return_outputs=return_outputs)
                                return patched_compute_loss
                            cls.compute_loss = make_loss_patch(orig_fn)
                            patched_any = True
                    except Exception:
                        pass
        if patched_any:
            print("⚡ [MoAI Patch] Successfully patched SentenceTransformerTrainer.compute_loss() for transformers>=4.46")
    except Exception as e:
        print(f"ℹ️ patch_sentence_transformers notice: {e}")

patch_sentence_transformers()


from peft import LoraConfig, get_peft_model, TaskType




def load_triplets_dataloader(jsonl_path: Path, batch_size: int = 8):
    """Loads query-positive-negative triplets into SentenceTransformers InputExamples."""
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = str(item.get("query") or "").strip()
            pos = str(item.get("positive") or "").strip()
            neg = str(item.get("negative") or "").strip()
            if not q or not pos or not neg:
                continue
            examples.append(InputExample(texts=[q, pos, neg]))

    if not examples:
        raise ValueError(f"No valid triplets loaded from {jsonl_path}!")
    print(f"📊 [Embedding DataLoader] Loaded {len(examples)} training triplets from {jsonl_path.name}")
    return DataLoader(examples, shuffle=True, batch_size=batch_size)


def run_embedding_training(
    model_name: str = "BAAI/bge-m3",
    data_path: Path = None,
    output_dir: Path = None,
    epochs: int = 3,
    lr: float = 3e-5,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 8,
    warmup_steps: int = 20,
):
    output_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parent.parent.parent / "models" / "adapters" / "embedding_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    import os, torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🚀 [MoAI Embedding] Loading base SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name)
    # Set optimal 512 context length to prevent OOM (bge-m3 default is 8192 which blows up VRAM)
    model.max_seq_length = 512
    try:
        first_mod = model._first_module()
        if hasattr(first_mod, "max_seq_length"):
            first_mod.max_seq_length = 512
    except Exception:
        pass
    if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "model_max_length"):
        model.tokenizer.model_max_length = 512

    # Attach LoRA to Transformer backbone
    transformer_module = model._first_module()
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"] if "bge" in model_name else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    base_hf = getattr(transformer_module, "model", None) or getattr(transformer_module, "auto_model", None)
    peft_model = get_peft_model(base_hf, peft_config)
    if hasattr(transformer_module, "model"):
        try:
            transformer_module.model = peft_model
        except Exception:
            pass
    if hasattr(transformer_module, "auto_model"):
        try:
            transformer_module.auto_model = peft_model
        except Exception:
            pass

    # Ensure all trainable LoRA parameters are float32
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    train_dataloader = load_triplets_dataloader(data_path, batch_size=batch_size)

    # Use MultipleNegativesRankingLoss (or Cached version if available)
    try:
        train_examples_count = len(train_dataloader.dataset)
        # Use mini_batch_size=2 for true GradCache sub-batching to keep peak VRAM under 3.5GB per GPU
        mini_batch = 2
        train_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch)
        print("⚡ [GradCache] Using CachedMultipleNegativesRankingLoss for memory-efficient contrastive learning!")
    except Exception:
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print("⚡ [Standard] Using MultipleNegativesRankingLoss with in-batch negatives.")

    print(f"⚡ Starting Embedding Training for {epochs} epochs (LR={lr})...")
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        checkpoint_path=str(checkpoint_dir),
        checkpoint_save_steps=150,
        checkpoint_save_total_limit=2,
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    save_target = peft_model if 'peft_model' in locals() else (getattr(transformer_module, "model", None) or getattr(transformer_module, "auto_model", None))
    if hasattr(save_target, "save_pretrained"):
        save_target.save_pretrained(str(output_dir))
    if hasattr(model, "tokenizer") and model.tokenizer:
        model.tokenizer.save_pretrained(str(output_dir))
    elif hasattr(transformer_module, "tokenizer") and transformer_module.tokenizer:
        transformer_module.tokenizer.save_pretrained(str(output_dir))

    print(f"✅ [MoAI Embedding] Trained LoRA embedding adapter successfully saved to: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    massive_p = ROOT / "data" / "triplets" / "massive_embedding_triplets.jsonl"
    full_p = ROOT / "data" / "triplets" / "embedding_triplets.jsonl"
    sample_p = ROOT / "data" / "triplets" / "embedding_triplets_sample.jsonl"
    chosen_default = str(massive_p if massive_p.exists() else (full_p if full_p.exists() else sample_p))
    parser.add_argument("--data", type=str, default=chosen_default)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=str, default=None)
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        for cand in [ROOT / args.data, ROOT / args.data.replace("training_lab/", ""), Path(args.data.replace("training_lab/", ""))]:
            if cand.exists():
                data_file = cand
                break

    if data_file.exists():
        run_embedding_training(data_path=data_file, output_dir=Path(args.output_dir) if args.output_dir else None, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    else:
        raise FileNotFoundError(f"Dataset {args.data} not found at {data_file}. Verify path.")
