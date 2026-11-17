# -*- coding: utf-8 -*-
"""
Hyperparameter & Hardware Configuration for Agent 4 (Fact-Checker 3-Class NLI Verifier)
Optimized for Kaggle Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM).
"""

from dataclasses import dataclass, field
from pathlib import Path

LABEL2ID = {
    "SUPPORT": 0, "SUPPORTS": 0, "ENTAILMENT": 0,
    "CONTRADICT": 1, "CONTRADICTION": 1, "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2, "NEUTRAL": 2
}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NOT_ENOUGH_INFO"}


@dataclass
class Agent4TrainingConfig:
    base_model_id: str = "BAAI/bge-reranker-v2-m3"
    fallback_model_id: str = "allenai/scibert_scivocab_uncased"
    dataset_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" / "nli_verifier.jsonl")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")

    epochs: int = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    label_smoothing_factor: float = 0.05
    class_weights: list[float] = field(default_factory=lambda: [1.0, 1.35, 1.15])  # Penalize missed CONTRADICT claims
    val_split_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
