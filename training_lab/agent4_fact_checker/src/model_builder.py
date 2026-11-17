# -*- coding: utf-8 -*-
"""
Cross-Encoder NLI Model Builder for Agent 4 (`BAAI/bge-reranker-v2-m3` or `allenai/scibert_scivocab_uncased`).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import Agent4TrainingConfig, LABEL2ID, ID2LABEL


def build_nli_cross_encoder(cfg: Agent4TrainingConfig):
    """Initializes 3-class SequenceClassification model (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_id,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        print(f"⚠️ [Agent 4] Failed to load {cfg.base_model_id}: {e}. Falling back to {cfg.fallback_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.fallback_model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.fallback_model_id,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    return model, tokenizer
