import os
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
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

logger = setup_logger("MotorRecomendacion-Trainer")

class RecommenderTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.epochs = int(config.get("epochs", 10))
        self.lr = float(config.get("learning_rate", 0.0002))
        self.output_dir = config.get("output_dir", "checkpoints/motor_recomendacion")
        self.drive_dir = config.get("drive_backup_dir", None)
        self.save_every = int(config.get("save_every", 2))

        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model.to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.best_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, total = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoca {epoch:02d}/{self.epochs:02d} [PolyVore]", leave=True, ncols=100)

        for item_a, item_b, target in pbar:
            item_a = item_a.to(self.device)
            item_b = item_b.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            score = self.model(item_a, item_b)
            loss = self.criterion(score, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * item_a.size(0)
            total += item_a.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(total, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, total = 0.0, 0
        for item_a, item_b, target in self.val_loader:
            item_a = item_a.to(self.device)
            item_b = item_b.to(self.device)
            target = target.to(self.device)

            score = self.model(item_a, item_b)
            loss = self.criterion(score, target)
            total_loss += loss.item() * item_a.size(0)
            total += item_a.size(0)

        return total_loss / max(total, 1)

    def fit(self):
        start_epoch = int(self.config.get("start_epoch", 1))
        if start_epoch > self.epochs:
            logger.info(f"[INFO] PolyVore ya completo la epoca {start_epoch - 1}. Para continuar, aumenta --epochs a mas de {start_epoch - 1}.")
            return

        logger.info(f"[TRAIN] Iniciando entrenamiento PolyVore: Epoca {start_epoch:02d} hasta {self.epochs:02d} en {self.device}...")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            logger.info(f"Epoca {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            checkpoint_state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config
            }
            ckpt_filename = f"recommender_epoch_{epoch:02d}.pt"
            latest_filename = "recommender_latest.pt"
            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, ckpt_filename), self.drive_dir, is_best=is_best)
            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, latest_filename), self.drive_dir, is_best=False)

        logger.info(f"[DONE] Entrenamiento finalizado. Mejor Val Loss: {self.best_loss:.4f}")
