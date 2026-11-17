# -*- coding: utf-8 -*-
"""
Hyperparameter & Hardware Configuration for Agent 3 (Academic Summarizer: Generator + Critic)
Optimized for Kaggle Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Agent3TrainingConfig:
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset")
    weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")

    # Generator QLoRA + DoRA (Stage 1 SFT + Stage 2 DPO)
    gen_lora_r: int = 64
    gen_lora_alpha: int = 128
    gen_epochs_sft: int = 3
    gen_epochs_dpo: int = 2
    gen_lr_sft: float = 2e-4
    gen_lr_dpo: float = 5e-6
    dpo_beta: float = 0.1
    max_seq_length: int = 1536

    # Critic QLoRA + DoRA (Adversarial Hard-Negatives)
    critic_lora_r: int = 32
    critic_lora_alpha: int = 64
    critic_epochs: int = 3
    critic_lr: float = 1.5e-4
    critic_max_seq_length: int = 1536

    # Common Dual T4 Settings
    lora_dropout: float = 0.05
    use_dora: bool = False
    neftune_noise_alpha: float = 5.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_memory_per_gpu: str = "14.0GiB"
    seed: int = 42

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        self.weights_dir = Path(self.weights_dir)
