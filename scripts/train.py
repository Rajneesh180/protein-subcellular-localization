"""Training entry point."""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import HPADataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.factory import build_model
from src.training.losses import FocalLoss, get_pos_weights
from src.training.trainer import Trainer


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

    # seed everything
    torch.manual_seed(train_cfg["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(train_cfg["seed"])

    # dataset
    image_size = cfg["data"]["image_size"]
    full_dataset = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_train_transforms(image_size),
        image_size=image_size,
    )

    # train/val split
    val_size = int(len(full_dataset) * cfg["data"]["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # apply val transforms to val split
    val_dataset.dataset = HPADataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["image_dir"],
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
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
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        pos_weight = get_pos_weights(cfg["data"]["train_csv"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg["scheduler_patience"],
        factor=train_cfg["scheduler_factor"],
    )

    # train
    trainer = Trainer(
        model=model,
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
