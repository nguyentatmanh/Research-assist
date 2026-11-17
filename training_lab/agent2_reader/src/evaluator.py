# -*- coding: utf-8 -*-
"""
Evaluator for Agent 2 (LayoutLMv3 Document Reader).
Computes token-level classification metrics: Accuracy, Macro F1, and Per-Class F1
across all 11 DocLayNet structural classes (Table, Formula, Section-header, Text, etc.).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from .config import DOCLAYNET_LABELS


def compute_layout_metrics(eval_pred):
    """Computes Macro F1, Accuracy, and per-class metrics ignoring pad tokens (-100)."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    preds = np.argmax(predictions, axis=2)
    labels = np.asarray(labels).squeeze()
    if labels.ndim == 1:
        labels = np.expand_dims(labels, 0)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, 0)

    # Flatten and filter out ignored index -100
    true_labels = []
    true_preds = []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                true_labels.append(l)
                true_preds.append(p)

    if not true_labels:
        res = {"accuracy": 0.0, "macro_f1": 0.0}
        for name in DOCLAYNET_LABELS:
            res[f"f1_{name.lower().replace('-', '_')}"] = 0.0
        return res

    acc = accuracy_score(true_labels, true_preds)
    macro_f1 = f1_score(true_labels, true_preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
    }

    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, true_preds, average=None, labels=list(range(len(DOCLAYNET_LABELS))), zero_division=0
    )
    for idx, name in enumerate(DOCLAYNET_LABELS):
        if idx < len(f1_per_class):
            metrics[f"f1_{name.lower().replace('-', '_')}"] = round(float(f1_per_class[idx]), 4)

    return metrics


def evaluate_layout_parser(model, tokenizer) -> dict:
    """Verifies model classification head and label map consistency."""
    num_classes = getattr(model.config, "num_labels", len(DOCLAYNET_LABELS))
    if num_classes != len(DOCLAYNET_LABELS):
        print(f"⚠️ [Agent 2 Evaluator] Warning: model num_labels ({num_classes}) != {len(DOCLAYNET_LABELS)}")
    else:
        print(f"✅ [Agent 2 Evaluator] Verified {num_classes} DocLayNet classes: {', '.join(DOCLAYNET_LABELS)}")
    return {"num_classes": num_classes, "labels": DOCLAYNET_LABELS, "status": "verified" if num_classes == len(DOCLAYNET_LABELS) else "mismatched"}
