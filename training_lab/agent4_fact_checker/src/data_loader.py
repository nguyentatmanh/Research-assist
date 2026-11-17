# -*- coding: utf-8 -*-
"""
Dataset Loader & Cross-Encoder Pair Tokenizer for Agent 4 (SciFact + VitaminC NLI Verifier).
Supports both (`premise`, `hypothesis`) and (`evidence`, `claim`) schema formats.
Removes unused text columns to prevent DataCollator tensor conflicts.
"""

import json
from pathlib import Path
from datasets import Dataset
from .config import LABEL2ID


def load_nli_dataset(dataset_path: Path, tokenizer, max_length: int = 512, val_ratio: float = 0.15, seed: int = 42):
    """Loads `nli_verifier.jsonl` and tokenizes premise/hypothesis pairs for Cross-Encoder NLI."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"NLI dataset not found at {dataset_path}")
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_premise = row.get("premise") or row.get("evidence") or row.get("document") or ""
            raw_hypothesis = row.get("hypothesis") or row.get("claim") or row.get("query") or ""
            premise = " ".join([str(p) for p in raw_premise]) if isinstance(raw_premise, (list, tuple)) else str(raw_premise).strip()
            hypothesis = " ".join([str(h) for h in raw_hypothesis]) if isinstance(raw_hypothesis, (list, tuple)) else str(raw_hypothesis).strip()
            if not premise or not hypothesis:
                continue

            if "label_id" in row:
                lbl = max(0, min(2, int(row["label_id"])))
            elif "label" in row and isinstance(row["label"], (int, float)):
                lbl = max(0, min(2, int(row["label"])))
            else:
                label_str = str(row.get("label", "NOT_ENOUGH_INFO")).upper()
                lbl = LABEL2ID.get(label_str, 2)

            records.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "labels": lbl,
            })

    if not records:
        raise ValueError(f"No valid NLI records loaded from {dataset_path}!")
    ds = Dataset.from_list(records)
    val_size = max(1, int(len(ds) * val_ratio)) if len(ds) > 1 else 0
    if len(ds) <= 1 or val_size == 0:
        split = {"train": ds, "test": ds}
    else:
        split = ds.train_test_split(test_size=val_size, seed=seed)

    def tokenize_fn(batch):
        return tokenizer(
            batch["premise"], 
            batch["hypothesis"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )

    train_ds = split["train"].map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    val_ds = split["test"].map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    print(f"📊 [Agent 4 DataLoader] Cleaned & Loaded {len(ds)} NLI pairs -> Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
