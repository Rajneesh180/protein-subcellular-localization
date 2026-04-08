"""Loss functions for multi-label classification."""

import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in multi-label setting.

    Alpha weights: alpha for positive samples, (1-alpha) for negatives.
    Higher alpha up-weights rare positive labels.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
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
    neg_counts = total - counts

    # log-damped weights to avoid extreme values for very rare classes
    weights = np.log1p(neg_counts / np.clip(counts, 1, None))
    weights = np.clip(weights, 0.5, 10.0)
    return torch.tensor(weights, dtype=torch.float32)
