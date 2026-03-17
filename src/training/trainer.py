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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.best_f1 = 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)

        # mlflow setup
        mlflow.set_experiment(experiment_name)

    def train_epoch(self) -> dict:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss

        return metrics

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

    def fit(self, epochs: int, freeze_epochs: int = 5, unfreeze_blocks: int = 2):
        """Full training loop with optional backbone freezing."""

        with mlflow.start_run():
            mlflow.log_params({
                "model": self.model.__class__.__name__,
                "epochs": epochs,
                "freeze_epochs": freeze_epochs,
                "lr": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
            })

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

                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                # log to mlflow
                for k, v in train_metrics.items():
                    mlflow.log_metric(f"train_{k}", v, step=epoch)
                for k, v in val_metrics.items():
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
                    path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "f1_macro": self.best_f1,
                    }, path)
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
