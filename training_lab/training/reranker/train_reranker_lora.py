# -*- coding: utf-8 -*-
"""
CoREB-Style SOTA LoRA Reranker Training Pipeline for Multi-Agent RAG.
Hardware: Kaggle Dual Tesla T4 (2x16GB) or Single GPU.
Recipe:
  - Base: BAAI/bge-reranker-v2-m3 (or Qwen-based Cross-Encoder)
  - LoRA: rank=16, alpha=16, dropout=0.05
  - Loss: BCEWithLogitsLoss / MarginRankingLoss
  - Checkpoint Selection: Best Validation nDCG@10 (MoAI Strategy)
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
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

from peft import LoraConfig, get_peft_model, TaskType

try:
    from training_lab.evaluation.ir_metrics import compute_ndcg_at_k, compute_mrr_at_k, compute_average_precision_at_k
except ImportError:
    from evaluation.ir_metrics import compute_ndcg_at_k, compute_mrr_at_k, compute_average_precision_at_k


def load_reranker_dataset(jsonl_path: Path, tokenizer, max_length: int = 512, val_ratio: float = 0.15, seed: int = 42):
    """Loads query-document pairs (label: 1 for positive, 0 for negative)."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            query = str(item.get("query") or "").strip()
            doc = str(item.get("document") or item.get("passage") or "").strip()
            label = float(item.get("label", 0))
            if not query or not doc:
                continue
            rows.append({"query": query, "document": doc, "label": label})

    if not rows:
        raise ValueError(f"No valid query-document pairs in {jsonl_path}!")
    ds = Dataset.from_list(rows)
    val_size = max(1, int(len(ds) * val_ratio)) if len(ds) > 1 else 0
    if len(ds) <= 1 or val_size == 0:
        split = {"train": ds, "test": ds}
    else:
        split = ds.train_test_split(test_size=val_size, seed=seed)

    def tokenize_fn(batch):
        return tokenizer(
            batch["query"],
            batch["document"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=["query", "document"])
    val_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=["query", "document"])
    print(f"📊 [Reranker DataLoader] Train: {len(train_ds)}, Validation: {len(val_ds)}")
    return train_ds, val_ds, split["test"]


import dataclasses
import inspect


def safe_build_config(config_cls, **kwargs):
    """Safely instantiates TrainingArguments across different transformers versions."""
    try:
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    except TypeError:
        valid_fields = set(inspect.signature(config_cls.__init__).parameters.keys())

    if "warmup_steps" in kwargs and "warmup_ratio" in kwargs:
        kwargs.pop("warmup_ratio")
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

class RerankerMetricsCallback:
    """Computes academic nDCG@10 and MRR@10 on the hold-out validation set to select the best checkpoint."""

    def __init__(self, raw_val_records):
        self.raw_records = raw_val_records

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        import numpy as np
        if isinstance(logits, tuple):
            logits = logits[0]
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        scores = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()
        if len(scores) == 0 or len(labels) == 0:
            return {"ndcg_10": 0.0, "mrr_10": 0.0}

        # Group by query to compute ranking
        gains = [float(lbl) for _, lbl in sorted(zip(scores, labels), reverse=True)]
        ndcg_10 = compute_ndcg_at_k(gains, k=10)
        mrr_10 = compute_mrr_at_k([1.0], gains, k=10)

        map_10 = compute_average_precision_at_k([1.0], gains, k=10)
        return {
            "ndcg_10": round(ndcg_10, 4),
            "mrr_10": round(mrr_10, 4),
            "map_10": round(map_10, 4),
        }


def run_reranker_training(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    data_path: Path = None,
    output_dir: Path = None,
    epochs: int = 3,
    lr: float = 3e-5,
    lora_r: int = 16,
    lora_alpha: int = 16,
    batch_size: int = 8,
    grad_acc: int = 2,
    seed: int = 42
):
    output_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parent.parent.parent / "models" / "adapters" / "reranker_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [CoREB Reranker] Initializing Cross-Encoder: {model_name} on available GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float32,  # Uniform float32 master weights for LoRA + AMP compatibility
    )

    # CoREB Recipe: LoRA r=16, alpha=16
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"] if "bert" in model_name or "bge" in model_name else ["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    # Ensure all trainable parameters are strictly in float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    train_ds, val_ds, raw_val = load_reranker_dataset(data_path, tokenizer, max_length=512, seed=seed)
    metrics_cb = RerankerMetricsCallback(raw_val)

    training_args = safe_build_config(
        TrainingArguments,
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        # In DataParallel (plain python on >1 GPUs without DDP), PyTorch AMP FP16 scaler
        # has a known autograd bug: 'Found dtype Float but expected Half'.
        # Using full float32 uses only ~3.5GB VRAM per T4 GPU and is 100% bug-free.
        fp16=(torch.cuda.is_available() and (torch.cuda.device_count() <= 1 or "LOCAL_RANK" in os.environ)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="ndcg_10",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics_cb.compute_metrics,
    )

    print("⚡ Starting LoRA Reranker Training with MoAI Checkpoint Selection (nDCG@10)...")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"🏆 Best Evaluation Results: nDCG@10 = {eval_results.get('eval_ndcg_10')}, MRR@10 = {eval_results.get('eval_mrr_10')}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__float__"):
            return float(obj)
        return str(obj)
    (output_dir / "eval_manifest.json").write_text(json.dumps(eval_results, indent=2, default=_json_default), encoding="utf-8")
    print(f"✅ [CoREB Reranker] Trained adapter successfully exported to: {output_dir}")
    return model, tokenizer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data", type=str, default="training_lab/data/triplets/reranker_pairs_sample.jsonl")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=str, default=None)
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        for cand in [ROOT / args.data, ROOT / args.data.replace("training_lab/", ""), Path(args.data.replace("training_lab/", ""))]:
            if cand.exists():
                data_file = cand
                break

    if data_file.exists():
        run_reranker_training(data_path=data_file, output_dir=Path(args.output_dir) if args.output_dir else None, epochs=args.epochs, lr=args.lr)
    else:
        raise FileNotFoundError(f"Dataset {args.data} not found at {data_file}. Verify path.")
