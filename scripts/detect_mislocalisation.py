"""Detect protein mislocalisation using trained models."""

import argparse
import os

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HPADataset
from src.data.augmentation import get_val_transforms
from src.models.factory import build_model
from src.evaluation.metrics import find_best_threshold
from src.analysis.mislocalisation import (
    MislocalisationDetector,
    visualize_mislocalisation_report,
    visualize_cooccurrence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect protein mislocalisation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["baseline", "cbam", "hybrid"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--threshold", type=float, default=None,
                        help="binarization threshold (auto if not set)")
    parser.add_argument("--anomaly-threshold", type=float, default=0.5,
                        help="anomaly score cutoff for flagging")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = (
        args.checkpoint
        or os.path.join("checkpoints", f"{args.model}_best.pth")
    )
    model_kwargs = {
        "num_classes": cfg["data"]["num_classes"],
        "pretrained": False,
        "drop_rate": cfg["model"]["drop_rate"],
    }
    if args.model == "cbam":
        model_kwargs["reduction"] = cfg["cbam"]["reduction_ratio"]
        model_kwargs["kernel_size"] = cfg["cbam"]["kernel_size"]
    elif args.model == "hybrid":
        model_kwargs["embed_dim"] = cfg["transformer"]["embed_dim"]
        model_kwargs["num_heads"] = cfg["transformer"]["num_heads"]
        model_kwargs["depth"] = cfg["transformer"]["depth"]
        model_kwargs["mlp_ratio"] = cfg["transformer"]["mlp_ratio"]

    model = build_model(args.model, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {args.model} from {checkpoint_path}")

    image_size = cfg["data"]["image_size"]
    full_dataset = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    # reproduce train/val split
    seed = cfg["training"]["seed"]
    n = len(full_dataset)
    val_size = int(n * cfg["data"]["val_split"])
    indices = list(range(n))
    label_sums = full_dataset.targets.sum(axis=1)
    sorted_indices = sorted(indices, key=lambda i: label_sums[i])
    k = max(1, n // val_size) if val_size > 0 else n + 1
    val_indices = sorted_indices[::k][:val_size]
    train_indices = [i for i in indices if i not in set(val_indices)]

    # build detector from training distribution
    train_targets = full_dataset.targets[train_indices]
    detector = MislocalisationDetector.from_training_data(train_targets)

    visualize_cooccurrence(
        detector.cooccurrence,
        save_path=os.path.join(args.output_dir, "cooccurrence_matrix.png"),
    )

    # inference on validation set
    val_dataset = Subset(full_dataset, val_indices)
    loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []
    all_images = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())
            all_images.append(images.cpu())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_images = torch.cat(all_images)

    if args.threshold:
        thresh = args.threshold
    else:
        thresh = find_best_threshold(all_targets, all_preds)
    print(f"Using threshold: {thresh:.2f}")

    binary_preds = (all_preds >= thresh).astype(int)

    results = detector.detect_batch(
        binary_preds, expected=all_targets.astype(int),
    )

    flagged = [r for r in results if r["anomaly_score"] > args.anomaly_threshold]
    print(f"\nMislocalisation detection results:")
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Flagged anomalous: {len(flagged)} ({100 * len(flagged) / len(results):.1f}%)")

    print(f"\nTop anomalies:")
    for r in results[:10]:
        idx = r["sample_idx"]
        print(f"  Sample {idx}: anomaly={r['anomaly_score']:.3f}")
        print(f"    Predicted: {', '.join(r['predicted_compartments']) or 'none'}")
        if r.get("expected_compartments"):
            print(f"    Expected:  {', '.join(r['expected_compartments']) or 'none'}")
        if r.get("unexpected_labels"):
            print(f"    UNEXPECTED: {', '.join(r['unexpected_labels'])}")

    visualize_mislocalisation_report(
        results,
        images=all_images,
        save_path=os.path.join(args.output_dir, "mislocalisation_report.png"),
        top_k=8,
    )

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
