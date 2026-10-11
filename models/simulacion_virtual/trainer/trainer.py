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

logger = setup_logger("SimulacionVirtual-Trainer")

class SimulacionTrainer:
    def __init__(
        self,
        model: nn.Module,
        task: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.epochs = int(config.get("epochs", 10))
        self.start_epoch = int(config.get("start_epoch", 1))
        self.lr = float(config.get("learning_rate", 0.0002))
        self.output_dir = config.get("output_dir", f"checkpoints/{task}")
        self.drive_dir = config.get("drive_backup_dir", None)
        self.save_every = int(config.get("save_every", 1))

        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model.to(self.device)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.best_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, total = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoca {epoch:02d}/{self.epochs:02d} [{self.task}]", leave=True, ncols=100)

        for img_in, img_target in pbar:
            img_in = img_in.to(self.device)
            img_target = img_target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(img_in)
            loss = self.criterion(out, img_target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * img_in.size(0)
            total += img_in.size(0)
            pbar.set_postfix({"L1_loss": f"{loss.item():.4f}"})

        return total_loss / max(total, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, total = 0.0, 0
        for img_in, img_target in self.val_loader:
            img_in = img_in.to(self.device)
            img_target = img_target.to(self.device)
            out = self.model(img_in)
            loss = self.criterion(out, img_target)
            total_loss += loss.item() * img_in.size(0)
            total += img_in.size(0)
        return total_loss / max(total, 1)

    def fit(self):
        if self.start_epoch > self.epochs:
            logger.info(f"[*] El entrenamiento ya alcanzo la epoca {self.start_epoch - 1} de {self.epochs}. Nada pendiente.")
            return

        logger.info(f"[TRAIN] Iniciando entrenamiento para '{self.task}' desde epoca {self.start_epoch} hasta {self.epochs} en {self.device}...")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            logger.info(f"Epoca {epoch:02d}/{self.epochs:02d} | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f}")

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            checkpoint_state = {
                "epoch": epoch,
                "task": self.task,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config
            }
            ckpt_filename = f"epoch_{epoch:02d}.pt"
            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, ckpt_filename), self.drive_dir, is_best=is_best)

            latest_filename = f"latest_model.pt"
            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, latest_filename), self.drive_dir, is_best=False)

        logger.info(f"[DONE] Entrenamiento finalizado para '{self.task}'. Mejor Val Loss: {self.best_loss:.4f}")
