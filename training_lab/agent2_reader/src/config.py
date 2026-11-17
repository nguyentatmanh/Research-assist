# -*- coding: utf-8 -*-
"""
Hyperparameter & Hardware Configuration for Agent 2 (Document Reader - LayoutLMv3)
Optimized for Kaggle Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM).
"""

from dataclasses import dataclass, field
from pathlib import Path

DOCLAYNET_LABELS = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]


@dataclass
class Agent2TrainingConfig:
    base_model_id: str = "microsoft/layoutlmv3-base"
    dataset_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" / "layout_doclaynet_sample.jsonl")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")

    epochs: int = 5
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 512
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    val_split_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
