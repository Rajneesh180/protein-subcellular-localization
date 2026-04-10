"""Protein mislocalisation detection via prediction-reference comparison.

Compares model predictions against expected subcellular localization patterns
to flag potentially mislocalised proteins. Useful for screening disease-related
or drug-induced changes in protein trafficking.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from src.data.dataset import HPA_LABELS, NUM_CLASSES


COMPARTMENT_GROUPS = {
    "nuclear": [0, 1, 2, 3, 4, 5],
    "secretory": [6, 7],
    "cytoskeletal": [8, 9, 10],
    "centrosome": [11, 12],
    "membrane": [13],
    "mitochondria": [14],
    "cytoplasmic": [15, 16, 17, 18, 19, 20, 21, 22],
    "negative": [23],
    "mitotic": [24, 25, 26, 27],
}

CLASS_TO_GROUP = {}
for _group, _indices in COMPARTMENT_GROUPS.items():
    for _idx in _indices:
        CLASS_TO_GROUP[_idx] = _group


class MislocalisationDetector:
    """Detect abnormal protein localisation by comparing predictions
    against reference co-occurrence patterns learned from training data."""

    def __init__(self, cooccurrence: np.ndarray, frequencies: np.ndarray):
        self.cooccurrence = cooccurrence
        self.frequencies = frequencies

    @classmethod
    def from_training_data(cls, targets: np.ndarray) -> "MislocalisationDetector":
        n_samples = targets.shape[0]
        cooccurrence = (targets.T @ targets) / n_samples
        frequencies = targets.mean(axis=0)
        return cls(cooccurrence, frequencies)

    def score_prediction(
        self,
        predicted_labels: np.ndarray,
        expected_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Score how anomalous a single prediction is.

        Args:
            predicted_labels: (28,) binary vector of predicted labels.
            expected_labels: (28,) binary vector of known/expected labels.
                If None, scores based solely on co-occurrence rarity.
        """
        pred_indices = np.where(predicted_labels == 1)[0]

        result = {
            "predicted_compartments": [HPA_LABELS[i] for i in pred_indices],
            "predicted_groups": list(set(
                CLASS_TO_GROUP.get(i, "unknown") for i in pred_indices
            )),
        }

        if expected_labels is not None:
            expected_indices = np.where(expected_labels == 1)[0]
            unexpected = set(pred_indices) - set(expected_indices)
            missing = set(expected_indices) - set(pred_indices)

            result["expected_compartments"] = [HPA_LABELS[i] for i in expected_indices]
            result["unexpected_labels"] = [HPA_LABELS[i] for i in unexpected]
            result["missing_labels"] = [HPA_LABELS[i] for i in missing]

            intersection = len(set(pred_indices) & set(expected_indices))
            union = len(set(pred_indices) | set(expected_indices))
            result["jaccard_similarity"] = intersection / union if union > 0 else 1.0
            result["deviation_score"] = 1.0 - result["jaccard_similarity"]

        # co-occurrence anomaly: how unusual is this label combination?
        if len(pred_indices) >= 2:
            pair_scores = []
            for i in range(len(pred_indices)):
                for j in range(i + 1, len(pred_indices)):
                    ci, cj = pred_indices[i], pred_indices[j]
                    pair_scores.append(self.cooccurrence[ci, cj])
            result["cooccurrence_score"] = float(np.mean(pair_scores))
            result["rarity_score"] = 1.0 - result["cooccurrence_score"]
        elif len(pred_indices) == 1:
            result["cooccurrence_score"] = float(self.frequencies[pred_indices[0]])
            result["rarity_score"] = 1.0 - result["cooccurrence_score"]
        else:
            result["cooccurrence_score"] = 0.0
            result["rarity_score"] = 1.0

        if expected_labels is not None:
            result["anomaly_score"] = (
                0.6 * result["deviation_score"] + 0.4 * result["rarity_score"]
            )
        else:
            result["anomaly_score"] = result["rarity_score"]

        result["is_mislocalised"] = result["anomaly_score"] > 0.5
        return result

    def detect_batch(
        self,
        predictions: np.ndarray,
        expected: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Run mislocalisation detection on a batch of predictions."""
        results = []
        for i in range(len(predictions)):
            exp = expected[i] if expected is not None else None
            result = self.score_prediction(predictions[i], exp)
            result["sample_idx"] = i
            results.append(result)

        results.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return results


def build_cooccurrence_matrix(targets: np.ndarray) -> np.ndarray:
    """Build normalised co-occurrence matrix from multi-label targets."""
    return (targets.T @ targets) / len(targets)


def visualize_mislocalisation_report(
    results: List[Dict],
    images: Optional[torch.Tensor] = None,
    save_path: str = "mislocalisation_report.png",
    top_k: int = 8,
):
    """Visualize top mislocalised samples with predicted vs expected labels."""
    flagged = [r for r in results if r["is_mislocalised"]][:top_k]

    if not flagged:
        print("No mislocalised samples detected.")
        return

    n = len(flagged)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, result in enumerate(flagged):
        idx = result["sample_idx"]

        if images is not None and idx < len(images):
            img = images[idx, 1].cpu().numpy()
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title(
                f"Sample {idx} (anomaly: {result['anomaly_score']:.2f})"
            )
        else:
            axes[i, 0].text(
                0.5, 0.5, f"Sample {idx}",
                ha="center", va="center", fontsize=14,
            )
            axes[i, 0].set_title(f"Anomaly: {result['anomaly_score']:.2f}")
        axes[i, 0].axis("off")

        text_lines = []
        pred_text = ", ".join(result["predicted_compartments"]) or "none"
        text_lines.append(f"Predicted: {pred_text}")
        if "expected_compartments" in result:
            exp_text = ", ".join(result["expected_compartments"]) or "none"
            text_lines.append(f"Expected:  {exp_text}")
        if result.get("unexpected_labels"):
            text_lines.append(f"UNEXPECTED: {', '.join(result['unexpected_labels'])}")
        if result.get("missing_labels"):
            text_lines.append(f"MISSING: {', '.join(result['missing_labels'])}")
        text_lines.append(f"Rarity: {result['rarity_score']:.2f}")

        axes[i, 1].text(
            0.05, 0.5, "\n".join(text_lines),
            fontsize=10, va="center", family="monospace", wrap=True,
        )
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Localisation Analysis")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mislocalisation report to {save_path}")


def visualize_cooccurrence(
    cooccurrence: np.ndarray,
    save_path: str = "cooccurrence.png",
):
    """Plot co-occurrence matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cooccurrence, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(HPA_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(HPA_LABELS, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved co-occurrence matrix to {save_path}")
