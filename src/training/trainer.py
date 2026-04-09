"""Training loop with MLflow tracking."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow

from src.evaluation.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "hpa-protein-localization",
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.best_f1 = 0.0
        # mixed precision (CUDA only — MPS doesn't support GradScaler)
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        os.makedirs(checkpoint_dir, exist_ok=True)
        mlflow.set_experiment(experiment_name)

    def train_epoch(self, warmup_steps: int = 0, base_lrs: list = None,
                    global_step: int = 0, accumulation_steps: int = 1) -> tuple[dict, int]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="Training")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # LR warmup
            if global_step < warmup_steps and base_lrs is not None:
                warmup_factor = (global_step + 1) / warmup_steps
                for pg, blr in zip(self.optimizer.param_groups, base_lrs):
                    pg["lr"] = blr * warmup_factor

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets) / accumulation_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets) / accumulation_steps
                loss.backward()
                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps * images.size(0)
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss

        return metrics, global_step

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss

        return metrics

    def fit(self, epochs: int, freeze_epochs: int = 5, unfreeze_blocks: int = 2,
            warmup_epochs: int = 1, accumulation_steps: int = 1):
        """Full training loop with optional backbone freezing and LR warmup."""

        with mlflow.start_run():
            mlflow.log_params({
                "model": self.model.__class__.__name__,
                "model_name": self.model_name,
                "epochs": epochs,
                "freeze_epochs": freeze_epochs,
                "warmup_epochs": warmup_epochs,
                "accumulation_steps": accumulation_steps,
                "lr": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
            })

            # LR warmup: linearly ramp up from 0 to target LR over warmup_epochs
            warmup_steps = warmup_epochs * len(self.train_loader)
            base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
            global_step = 0

            # freeze backbone initially
            if hasattr(self.model, "freeze_backbone") and freeze_epochs > 0:
                self.model.freeze_backbone()
                print(f"Backbone frozen for first {freeze_epochs} epochs")

            patience_counter = 0
            early_stop_patience = 7

            for epoch in range(epochs):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'='*60}")

                # unfreeze after freeze phase
                if epoch == freeze_epochs and hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone(num_blocks=unfreeze_blocks)
                    print(f"Unfroze last {unfreeze_blocks} backbone blocks")

                train_metrics, global_step = self.train_epoch(
                    warmup_steps=warmup_steps,
                    base_lrs=base_lrs,
                    global_step=global_step,
                    accumulation_steps=accumulation_steps,
                )
                val_metrics = self.validate()

                # log to mlflow (skip non-scalar values like per-class lists)
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"train_{k}", v, step=epoch)
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"val_{k}", v, step=epoch)

                print(f"Train — loss: {train_metrics['loss']:.4f}, "
                      f"f1_macro: {train_metrics['f1_macro']:.4f}")
                print(f"Val   — loss: {val_metrics['loss']:.4f}, "
                      f"f1_macro: {val_metrics['f1_macro']:.4f}, "
                      f"roc_auc: {val_metrics.get('roc_auc_macro', 0):.4f}")

                # learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    mlflow.log_metric("lr", current_lr, step=epoch)

                # save best model
                if val_metrics["f1_macro"] > self.best_f1:
                    self.best_f1 = val_metrics["f1_macro"]
                    path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
                    ckpt = {
                        "epoch": epoch,
                        "model_name": self.model_name,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "f1_macro": self.best_f1,
                    }
                    if self.scheduler is not None:
                        ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
                    if self.scaler is not None:
                        ckpt["scaler_state_dict"] = self.scaler.state_dict()
                    ckpt["rng_state"] = torch.random.get_rng_state()
                    torch.save(ckpt, path)
                    print(f"Saved best model (f1={self.best_f1:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # early stopping
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # log best metric
            mlflow.log_metric("best_f1_macro", self.best_f1)
            print(f"\nTraining complete. Best F1-macro: {self.best_f1:.4f}")
