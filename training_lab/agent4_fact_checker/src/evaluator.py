# -*- coding: utf-8 -*-
"""
Evaluator & Temperature Scaling Calibration (ECE) for Agent 4 (Fact-Checker NLI Verifier).
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def compute_nli_metrics(eval_pred):
    """Computes Accuracy, Macro-F1, and Per-Class F1 (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).flatten()
    valid_mask = (labels >= 0) & (labels < 3)
    if not np.any(valid_mask):
        return {"accuracy": 0.0, "macro_f1": 0.0, "f1_support": 0.0, "f1_contradict": 0.0, "f1_nei": 0.0}
    clean_labels = labels[valid_mask]
    preds = np.argmax(logits, axis=-1).flatten()[valid_mask]
    acc = accuracy_score(clean_labels, preds)
    macro_f1 = f1_score(clean_labels, preds, average="macro", zero_division=0)
    _, _, f1_per_class, _ = precision_recall_fscore_support(clean_labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "f1_support": round(float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0, 4),
        "f1_contradict": round(float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0, 4),
        "f1_nei": round(float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0, 4),
    }


def calibrate_temperature_ece(logits: np.ndarray, labels: np.ndarray) -> float:
    """Finds optimal Temperature T in [0.5, 3.0] minimizing Negative Log-Likelihood for calibrated confidence."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).flatten()
    if len(labels) == 0 or len(logits) != len(labels):
        return 1.0
    valid_mask = (labels >= 0) & (labels < logits.shape[-1])
    if not np.any(valid_mask):
        return 1.0
    val_labels = labels[valid_mask].astype(int)
    val_logits = logits[valid_mask]

    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 2.5, 41):
        scaled = val_logits / t
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        nll = -np.mean(np.log(np.clip(probs[np.arange(len(val_labels)), val_labels], 1e-9, 1.0)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 3)
