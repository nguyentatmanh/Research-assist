# -*- coding: utf-8 -*-
"""
Dataset Loader for Agent 5 (Academic Report Writer SFT & RPO/DPO Preference Alignment).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_writer_sft_dataset(jsonl_path, tokenizer, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `writer_sft.jsonl` scholarly synthesis samples and applies ChatML formatting."""
    jsonl_path = Path(jsonl_path)
    rows = []
    if not jsonl_path.exists(): raise FileNotFoundError(f"{jsonl_path} does not exist")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            msgs = item.get("messages", [])
            if not msgs or len(msgs) < 2 or msgs[-1].get("role") != "assistant": continue
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
            rows.append({"text": text})
    ds = Dataset.from_list(rows)
    val_size = max(1, int(len(ds) * val_ratio)) if len(ds) > 1 else 0
    split = ds.train_test_split(test_size=val_size, seed=seed) if len(ds) > 1 and val_size > 0 else {"train": ds, "test": ds}
    print(f"📊 [Agent 5 SFT DataLoader] Loaded {len(ds)} scholarly synthesis samples -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]


def load_writer_dpo_dataset(jsonl_path, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `writer_dpo.jsonl` preference pairs (structured IEEE/Markdown with exact citations vs uncited text)."""
    jsonl_path = Path(jsonl_path)
    rows = []
    if not jsonl_path.exists(): raise FileNotFoundError(f"{jsonl_path} does not exist")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if not all(k in item for k in ("prompt", "chosen", "rejected")) or not str(item["prompt"]).strip():
                continue
            if str(item["chosen"]).strip() == str(item["rejected"]).strip():
                continue
            rows.append(item)
    ds = Dataset.from_list(rows)
    val_size = max(1, int(len(ds) * val_ratio)) if len(ds) > 1 else 0
    split = ds.train_test_split(test_size=val_size, seed=seed) if len(ds) > 1 and val_size > 0 else {"train": ds, "test": ds}
    print(f"📊 [Agent 5 DPO DataLoader] Loaded {len(ds)} citation-alignment pairs -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
