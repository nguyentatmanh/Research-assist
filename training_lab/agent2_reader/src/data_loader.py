# -*- coding: utf-8 -*-
"""
Dataset Loader & Spatial Bounding-Box Tokenizer for Agent 2 (LayoutLMv3 Document Reader).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_doclaynet_dataset(dataset_path: Path, tokenizer, max_length: int = 512, val_split_ratio: float = 0.1, seed: int = 42):
    """Loads DocLayNet JSONL samples (tokens, bboxes, ner_tags) and encodes into tensors for LayoutLMv3."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"DocLayNet dataset not found at {dataset_path}")
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens", [])
            raw_boxes = item.get("bboxes", [])
            tags = item.get("ner_tags", [])
            if not tokens or not raw_boxes or not tags:
                continue
            # Validate each bbox has exactly 4 coords
            bboxes = []
            valid_tokens = []
            valid_tags = []
            for t_idx, (tok, b, tag) in enumerate(zip(tokens, raw_boxes, tags)):
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    clamped = [max(0, min(1000, int(c))) for c in b]
                    bboxes.append(clamped)
                    valid_tokens.append(tok)
                    valid_tags.append(max(0, min(10, int(tag))) if isinstance(tag, (int, float)) else 9)
            if not valid_tokens:
                continue
            tokens, tags = valid_tokens, valid_tags
            enc = tokenizer(
                tokens,
                boxes=bboxes,
                word_labels=tags,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            rows.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "bbox": enc["bbox"],
                "labels": enc["labels"],
            })

    if not rows:
        raise ValueError(f"No valid DocLayNet samples found in {dataset_path}!")
    ds = Dataset.from_list(rows)
    val_size = max(1, int(len(ds) * val_split_ratio)) if len(ds) > 1 else 0
    if len(ds) <= 1 or val_size == 0:
        return ds, ds
    split = ds.train_test_split(test_size=val_size, seed=seed)
    print(f"📊 [Agent 2 DataLoader] Encoded {len(ds)} LayoutLMv3 pages -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
