# -*- coding: utf-8 -*-
"""
Hyperparameter & Hardware Configuration for Agent 1 (Retrieval Router & Query Planner)
Optimized for Kaggle Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Agent1TrainingConfig:
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" / "router_sft.jsonl")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")

    # QLoRA + DoRA Hyperparameters (Maximized for 32GB Dual T4)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_dora: bool = False
    use_rslora: bool = True  # Rank-Stabilized LoRA scaling (alpha / sqrt(r))
    attn_implementation: str = "sdpa"  # Optimal Memory-Efficient Attention for Tesla T4 (sm_75)
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training Schedule
    epochs: int = 4
    learning_rate: float = 1.5e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective Batch Size = 32 across Dual T4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    neftune_noise_alpha: float = 5.0
    val_split_ratio: float = 0.1
    seed: int = 42

    # Dual T4 VRAM Allocation
    max_memory_per_gpu: str = "14.0GiB"

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
