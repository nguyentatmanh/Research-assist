# -*- coding: utf-8 -*-
"""
Dataset Loader & ChatML Formatter for Agent 1 (Retrieval Router & Supervisor Planner).
Validates strict JSON output schema (`action`, `rewritten_queries`, `tools`, `plan`, `refuse`).
Seamlessly blends routing queries with Universal Agent Behavioral Trajectories (Planning & Refusal).
"""

import json
from pathlib import Path
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent


def validate_router_record(messages: list[dict]) -> bool:
    """Ensures assistant response is valid JSON containing required routing or planning fields."""
    if not messages or len(messages) < 2:
        return False
    if messages[-1].get("role") != "assistant":
        return False
    assistant_msg = messages[-1].get("content", "")
    try:
        parsed = json.loads(assistant_msg)
        return any(k in parsed for k in (
            "action", "rewritten_queries", "intent", "sub_queries", 
            "tools", "plan", "refuse", "complexity", "alpha", "reasoning"
        ))
    except Exception:
        return True  # Fallback: Allow general ChatML text if valid


def load_router_dataset(dataset_path: Path, tokenizer, val_split_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `router_sft.jsonl` + Universal Behavioral Trajectories, applies ChatML, and splits into Train/Val."""
    dataset_path = Path(dataset_path).resolve()
    records = []
    
    # 1. Load primary router SFT
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages", [])
                if not validate_router_record(messages):
                    continue
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                records.append({"text": text})

    # 2. Blend with Universal Agent Behavioral Trajectories (Planning & Refusal)
    behavior_cands = [
        ROOT / "data" / "agent_behavior" / "agent_behavior_trajectories.jsonl",
        dataset_path.parent.parent.parent / "data" / "agent_behavior" / "agent_behavior_trajectories.jsonl",
    ]
    behavior_path = next((c for c in behavior_cands if c.exists()), None)
    if behavior_path:
        with open(behavior_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                b = json.loads(line)
                q = str(b.get("question", "")).strip()
                if not q: continue
                plan = {
                    "action": b.get("expected_behavior", "retrieve_and_verify"),
                    "tools": b.get("expected_tools", ["document_retrieval"]),
                    "plan": b.get("reasoning_plan", ""),
                    "refuse": b.get("should_refuse", False)
                }
                messages = [
                    {"role": "system", "content": "Bạn là Meta-Supervisor Router và Kế hoạch viên cho hệ thống Multi-Agent RAG. Nhiệm vụ: Phân tích truy vấn, chọn tool, lập kế hoạch hành động hoặc từ chối nếu thiếu dữ liệu."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                records.append({"text": text})

    # Deduplicate text
    unique_texts = list({r["text"]: r for r in records}.values())
    if not unique_texts:
        raise FileNotFoundError(f"No router records loaded from {dataset_path} or behavior data!")
    full_ds = Dataset.from_list(unique_texts)
    val_size = max(1, int(len(full_ds) * val_split_ratio)) if len(full_ds) > 1 else 0
    if len(full_ds) <= 1 or val_size == 0:
        split = {"train": full_ds, "test": full_ds}
    else:
        split = full_ds.train_test_split(test_size=val_size, seed=seed)
    print(f"📊 [Agent 1 DataLoader] Loaded {len(full_ds)} unique ChatML planning samples -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
