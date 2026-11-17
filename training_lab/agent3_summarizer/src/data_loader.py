# -*- coding: utf-8 -*-
"""
Dataset Loader for Agent 3 (Generator SFT, Generator DPO Preference Pairs, and Critic Hard-Negatives).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_sft_jsonl_dataset(jsonl_path, tokenizer, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads ChatML JSONL (`generator_sft.jsonl` or `critic_sft.jsonl`) and splits into Train/Val."""
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
    print(f"📊 [Agent 3 SFT DataLoader] {jsonl_path.name}: {len(ds)} total -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]


def load_dpo_jsonl_dataset(jsonl_path, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads DPO Preference Pairs (`prompt`, `chosen`, `rejected`) from `generator_dpo.jsonl`."""
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
    print(f"📊 [Agent 3 DPO DataLoader] {jsonl_path.name}: {len(ds)} pairs -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
