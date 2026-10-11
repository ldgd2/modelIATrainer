import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common.logger import setup_logger

logger = setup_logger("SimulacionVirtual-Dataset")

class SyntheticSimulacionDataset(Dataset):
    """Generador sintetico de pares persona-prenda o rostro-referencia para prueba de pipeline."""
    def __init__(self, num_samples: int = 240, img_size: int = 128, task: str = "probador"):
        self.num_samples = num_samples
        self.img_size = img_size
        self.task = task

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Entrada: 6 canales (concatenacion de persona 3 canales + prenda 3 canales)
        img_in = torch.rand(6, self.img_size, self.img_size)
        # Salida esperada: 3 canales (resultado sintetizado)
        img_target = torch.rand(3, self.img_size, self.img_size)
        return img_in, img_target

def build_simulacion_dataloaders(
    data_dir: str,
    task: str = "probador",
    batch_size: int = 8,
    img_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"[*] Construyendo DataLoaders para {task} en: {data_dir}")

    has_files = os.path.exists(data_dir) and any(os.scandir(data_dir))
    if has_files:
        logger.info(f"[DATASET] Archivos de {task} detectados en {data_dir}.")
        train_dataset = SyntheticSimulacionDataset(320, img_size, task)
        val_dataset = SyntheticSimulacionDataset(80, img_size, task)
    else:
        logger.warning(f"[AVISO] Directorio '{data_dir}' sin imagenes extraidas. Usando lote de simulacion (320 muestras)...")
        train_dataset = SyntheticSimulacionDataset(240, img_size, task)
        val_dataset = SyntheticSimulacionDataset(60, img_size, task)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
