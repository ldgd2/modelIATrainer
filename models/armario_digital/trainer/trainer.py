import os
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            pass
from common.logger import setup_logger
from common.storage import save_checkpoint

logger = setup_logger("Armario-Trainer")

class ArmarioTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        self.config = config
        self.epochs = int(config.get("epochs", 20))
        self.lr = float(config.get("learning_rate", 0.0003))
        self.output_dir = config.get("output_dir", "checkpoints/armario")
        self.drive_dir = config.get("drive_backup_dir", None)
        self.patience = int(config.get("early_stopping_patience", 7))

        # Device selection
        cfg_dev = config.get("device", "auto")
        if cfg_dev == "cuda" or (cfg_dev == "auto" and torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"[DEVICE] Using compute device: {self.device}")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=float(config.get("weight_decay", 1e-4))
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoca {epoch:02d}/{self.epochs:02d} [Entrenamiento]", leave=True, ncols=100)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / max(total, 1)) * 100:.1f}%"
            })

        return total_loss / max(total, 1), (correct / max(total, 1)) * 100.0

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        val_loss = total_loss / max(total, 1)
        val_acc = (correct / max(total, 1)) * 100.0
        return val_loss, val_acc

    def fit(self):
        logger.info(f"[TRAIN] Starting training for {self.epochs} epochs...")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%"
            )

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "config": self.config
            }

            latest_path = os.path.join(self.output_dir, "latest_model.pt")
            save_checkpoint(checkpoint_state, latest_path, self.drive_dir, is_best=is_best)

            if self.patience_counter >= self.patience:
                logger.info(f"[STOP] Early stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs).")
                break

        logger.info(f"[DONE] Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
