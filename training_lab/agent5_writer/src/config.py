# -*- coding: utf-8 -*-
"""
Hyperparameter & Hardware Configuration for Agent 5 (Academic Report Writer & Citation Synthesizer)
SOTA Architecture: Qwen2.5-7B-Instruct + DoRA + rsLoRA + SDPA + Two-Stage SFT & RPO/DPO Alignment
Optimized for Kaggle Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Agent5TrainingConfig:
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")

    # SOTA DoRA + rsLoRA Hyperparameters
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_dora: bool = False
    use_rslora: bool = True
    attn_implementation: str = "sdpa"

    # Stage 1 SFT & Stage 2 RPO/DPO Schedule
    epochs_sft: int = 3
    max_steps: int = -1
    epochs_dpo: int = 2
    lr_sft: float = 1.5e-4
    lr_dpo: float = 5e-6
    dpo_beta: float = 0.1
    rpo_alpha: float = 0.1  # Regularized Preference Optimization (DPO + NLL stability)
    max_seq_length: int = 1536
    neftune_noise_alpha: float = 5.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_memory_per_gpu: str = "14.0GiB"
    seed: int = 42

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
