"""Evaluate trained models and generate visualizations."""

import argparse
import os
import random
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HPADataset, HPA_LABELS
from src.data.augmentation import get_val_transforms
from src.models.factory import build_model
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.evaluation.gradcam import generate_gradcam, visualize_gradcam_grid, visualize_attention_maps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint (default: checkpoints/{model}_best.pth)")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "cbam", "hybrid"])
    parser.add_argument("--gradcam", action="store_true", help="generate Grad-CAM visualizations")
    parser.add_argument("--attention", action="store_true", help="generate attention maps (hybrid only)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(model_name, checkpoint_path, cfg, device):
    model_kwargs = {
        "num_classes": cfg["data"]["num_classes"],
        "pretrained": False,
        "drop_rate": cfg["model"]["drop_rate"],
    }
    if model_name == "cbam":
        model_kwargs["reduction"] = cfg["cbam"]["reduction_ratio"]
        model_kwargs["kernel_size"] = cfg["cbam"]["kernel_size"]
    elif model_name == "hybrid":
        model_kwargs["embed_dim"] = cfg["transformer"]["embed_dim"]
        model_kwargs["num_heads"] = cfg["transformer"]["num_heads"]
        model_kwargs["depth"] = cfg["transformer"]["depth"]
        model_kwargs["mlp_ratio"] = cfg["transformer"]["mlp_ratio"]

    model = build_model(model_name, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def print_results(metrics, model_name):
    print(f"\n{'='*50}")
    print(f"Results — {model_name}")
    print(f"{'='*50}")
    print(f"F1-macro:        {metrics['f1_macro']:.4f}")
    print(f"F1-micro:        {metrics['f1_micro']:.4f}")
    print(f"Precision-macro: {metrics['precision_macro']:.4f}")
    print(f"Recall-macro:    {metrics['recall_macro']:.4f}")
    print(f"ROC-AUC-macro:   {metrics.get('roc_auc_macro', 0):.4f}")

    print(f"\nPer-class F1:")
    per_class = metrics.get("per_class_f1", [])
    for i, f1 in enumerate(per_class):
        print(f"  {HPA_LABELS[i]:35s} {f1:.4f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    checkpoint_path = args.checkpoint or os.path.join("checkpoints", f"{args.model}_best.pth")
    model = load_model(args.model, checkpoint_path, cfg, device)
    print(f"Loaded {args.model} from {checkpoint_path}")

    # dataset — use the same val split as training for fair evaluation
    image_size = cfg["data"]["image_size"]
    full_dataset = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    # reproduce the train/val split from train.py
    seed = cfg["training"]["seed"]
    n = len(full_dataset)
    val_size = int(n * cfg["data"]["val_split"])
    indices = list(range(n))
    label_sums = full_dataset.targets.sum(axis=1)
    sorted_indices = sorted(indices, key=lambda i: label_sums[i])
    k = max(1, n // val_size) if val_size > 0 else n + 1
    val_indices = sorted_indices[::k][:val_size]

    dataset = Subset(full_dataset, val_indices)
    print(f"Evaluating on {len(dataset)} validation samples")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # run inference
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
            if len(all_images) < 2:  # keep a few for viz
                all_images.append(images.cpu())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # find optimal threshold
    best_thresh = find_best_threshold(all_targets, all_preds)
    print(f"Optimal threshold: {best_thresh:.2f}")

    # compute metrics
    metrics = compute_metrics(all_targets, all_preds, threshold=best_thresh)
    print_results(metrics, args.model)

    # Grad-CAM
    if args.gradcam:
        sample_images = torch.cat(all_images[:1])[:8]

        # get target layer (last conv for efficientnet)
        if hasattr(model, "backbone"):
            target_layer = model.backbone.blocks[-1]
        else:
            target_layer = list(model.children())[-3]

        heatmaps = generate_gradcam(model, sample_images, target_layer, device=device)

        with torch.no_grad():
            sample_logits = model(sample_images.to(device))
            sample_preds = torch.sigmoid(sample_logits).cpu().numpy()

        sample_targets = all_targets[:8]
        save_path = os.path.join(args.output_dir, f"gradcam_{args.model}.png")
        visualize_gradcam_grid(sample_images, heatmaps, sample_preds, sample_targets, save_path)

    # Transformer attention maps
    if args.attention and args.model == "hybrid":
        sample_images = torch.cat(all_images[:1])[:4]
        save_path = os.path.join(args.output_dir, f"attention_{args.model}.png")
        visualize_attention_maps(model, sample_images, save_path, device)


if __name__ == "__main__":
    main()
