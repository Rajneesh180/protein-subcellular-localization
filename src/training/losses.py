"""Loss functions for multi-label classification."""

import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in multi-label setting.

    Reduces loss for well-classified examples, focusing on hard negatives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


def get_pos_weights(csv_path: str, num_classes: int = 28) -> torch.Tensor:
    """Compute positive class weights from label distribution for BCEWithLogitsLoss."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    counts = np.zeros(num_classes, dtype=np.float64)
    for targets in df["Target"]:
        for lbl in str(targets).split():
            counts[int(lbl)] += 1

    total = len(df)
    # weight = num_negatives / num_positives per class
    neg_counts = total - counts
    weights = neg_counts / np.clip(counts, 1, None)

    # clamp to avoid extreme weights for rare classes
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32)
