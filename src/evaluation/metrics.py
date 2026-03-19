"""Evaluation metrics for multi-label classification."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute multi-label classification metrics.

    Args:
        targets: (N, 28) binary ground truth
        predictions: (N, 28) predicted probabilities
        threshold: binarization threshold for predictions
    """
    binary_preds = (predictions >= threshold).astype(int)

    metrics = {}

    # macro-averaged metrics
    metrics["f1_macro"] = f1_score(targets, binary_preds, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(targets, binary_preds, average="micro", zero_division=0)
    metrics["precision_macro"] = precision_score(targets, binary_preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(targets, binary_preds, average="macro", zero_division=0)

    # per-class F1
    per_class_f1 = f1_score(targets, binary_preds, average=None, zero_division=0)
    metrics["per_class_f1"] = per_class_f1.tolist()

    # ROC-AUC (needs at least 2 classes present in targets)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(targets, predictions, average="macro")
        metrics["roc_auc_per_class"] = roc_auc_score(targets, predictions, average=None).tolist()
    except ValueError:
        # some classes might have 0 positive samples in validation split
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_per_class"] = []

    return metrics


def find_best_threshold(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Grid search for optimal binarization threshold based on F1-macro."""
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.7, 0.05):
        binary = (predictions >= thresh).astype(int)
        f1 = f1_score(targets, binary, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh
