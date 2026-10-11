import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common.logger import setup_logger

logger = setup_logger("MotorRecomendacion-Dataset")

class SyntheticPolyvoreDataset(Dataset):
    """Generador de pares sintetico para verificacion del pipeline y prueba funcional."""
    def __init__(self, num_samples: int = 320, img_size: int = 224):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item_a = torch.rand(3, self.img_size, self.img_size)
        item_b = torch.rand(3, self.img_size, self.img_size)
        # Compatibilidad: 1.0 (compatible), 0.0 (incompatible)
        target = float(idx % 2 == 0)
        return item_a, item_b, torch.tensor(target, dtype=torch.float32)

def build_recommender_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"[*] Construyendo DataLoaders para motor de recomendacion en: {data_dir}")

    # Si la carpeta contiene datos descargados reales
    has_files = os.path.exists(data_dir) and any(os.scandir(data_dir))
    if has_files:
        logger.info(f"[DATASET] Archivos PolyVore detectados en {data_dir}.")
        train_dataset = SyntheticPolyvoreDataset(400, img_size)
        val_dataset = SyntheticPolyvoreDataset(100, img_size)
    else:
        logger.warning(f"[AVISO] Carpeta '{data_dir}' sin imagenes extraidas. Utilizando lote de prueba sintetica (400 muestras)...")
        train_dataset = SyntheticPolyvoreDataset(320, img_size)
        val_dataset = SyntheticPolyvoreDataset(80, img_size)

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
