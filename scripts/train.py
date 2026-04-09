"""Training entry point."""

import argparse
import os
import random

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HPADataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.factory import build_model
from src.training.losses import FocalLoss, get_pos_weights
from src.training.trainer import Trainer

_SEED = 42

def _worker_init_fn(worker_id):
    np.random.seed(_SEED + worker_id)
    random.seed(_SEED + worker_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Train protein localization model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "cbam", "hybrid"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # override config with CLI args
    train_cfg = cfg["training"]
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything for reproducibility
    seed = train_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # datasets with separate instances for train/val (avoids Subset leakage)
    image_size = cfg["data"]["image_size"]
    train_full = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_train_transforms(image_size),
        image_size=image_size,
    )
    val_full = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    # stratified train/val split (preserves label distribution)
    n = len(train_full)
    val_size = int(n * cfg["data"]["val_split"])
    indices = list(range(n))
    # use label frequency as stratification key
    label_sums = train_full.targets.sum(axis=1)
    sorted_indices = sorted(indices, key=lambda i: label_sums[i])
    # interleave: put every k-th sample into val
    k = max(1, n // val_size) if val_size > 0 else n + 1
    val_indices = sorted_indices[::k][:val_size]
    val_set = set(val_indices)
    train_indices = [i for i in indices if i not in val_set]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    # reproducible DataLoader workers
    global _SEED
    _SEED = seed
    g = torch.Generator()
    g.manual_seed(seed)

    num_workers = train_cfg["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=_worker_init_fn,
    )

    # model
    model_kwargs = {
        "num_classes": cfg["data"]["num_classes"],
        "pretrained": cfg["model"]["pretrained"],
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
    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # loss
    if args.loss == "focal":
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        pos_weight = get_pos_weights(cfg["data"]["train_csv"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # optimizer with differential LR (lower for backbone, higher for head)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    lr = float(train_cfg["lr"])
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg["scheduler_patience"],
        factor=train_cfg["scheduler_factor"],
    )

    # train
    trainer = Trainer(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir="checkpoints",
        experiment_name=cfg["mlflow"]["experiment_name"],
    )

    trainer.fit(
        epochs=train_cfg["epochs"],
        freeze_epochs=train_cfg["freeze_epochs"],
        unfreeze_blocks=train_cfg["unfreeze_blocks"],
    )


if __name__ == "__main__":
    main()
