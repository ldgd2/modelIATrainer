import os
from typing import Dict, Any, Optional
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

logger = setup_logger("Asistente-Trainer")

class LLMTrainer:
    """
    Entrenador de Fine-Tuning para el Asistente Conversacional BellAI.
    Ejecuta optimizacion autorregresiva de instrucciones de moda sobre GPU.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Dict[str, Any],
        tokenizer = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.tokenizer = tokenizer

        self.epochs = int(config.get("epochs", 3))
        self.start_epoch = int(config.get("start_epoch", 1))
        self.lr = float(config.get("learning_rate", 2e-4))
        self.output_dir = config.get("output_dir", "checkpoints")
        self.drive_dir = config.get("drive_backup_dir", None)

        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model.to(self.device)

        # Optimizador AdamW estandar para LLMs
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=1e-5
        )

        self.best_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step_count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoca {epoch:02d}/{self.epochs:02d} [LLM-Stylist]",
            leave=True,
            ncols=100
        )

        for batch in pbar:
            self.optimizer.zero_grad()

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                # Si es modelo HuggingFace (CausalLM)
                if hasattr(self.model, "forward"):
                    try:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    except TypeError:
                        # Fallback a llamada simple
                        logits = self.model(input_ids)
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                input_ids = batch.to(self.device)
                outputs = self.model(input_ids)
                loss = outputs.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(step_count, 1)

    def fit(self):
        if self.start_epoch > self.epochs:
            logger.info(f"[*] El modelo ya completo la epoca {self.start_epoch - 1} de {self.epochs}. Nada pendiente.")
            return

        logger.info(f"[TRAIN] Iniciando Fine-Tuning de Asistente BellAI: Epoca {self.start_epoch:02d} hasta {self.epochs:02d} en {self.device}...")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.epochs + 1):
            loss = self.train_epoch(epoch)
            self.scheduler.step()

            logger.info(f"Epoca {epoch:02d}/{self.epochs:02d} | Train Loss: {loss:.4f}")

            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss

            # Guardar checkpoint numerado y puntero al ultimo modelo
            checkpoint_state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": loss,
                "config": self.config
            }

            ckpt_filename = f"stylist_epoch_{epoch:02d}.pt"
            latest_filename = "stylist_latest.pt"

            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, ckpt_filename), self.drive_dir, is_best=is_best)
            save_checkpoint(checkpoint_state, os.path.join(self.output_dir, latest_filename), self.drive_dir, is_best=False)

        logger.info(f"[DONE] Fine-Tuning completado exitosamente. Mejor Loss: {self.best_loss:.4f}")
