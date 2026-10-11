import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from models.armario_digital.dataset.transforms import get_armario_transforms
from common.logger import setup_logger

logger = setup_logger("Armario-Dataset")

class SyntheticArmarioDataset(Dataset):
    """Fallback dataset for testing pipeline integrity when real dataset is not yet present."""
    def __init__(self, num_samples: int = 40, num_classes: int = 10, img_size: int = 224, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Fake image tensor and label
        img = torch.rand(3, self.img_size, self.img_size)
        label = idx % self.num_classes
        return img, label

def build_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 2,
    num_classes: int = 10
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Creates train and validation DataLoaders.
    If data_dir contains subfolders with images, uses ImageFolder.
    Otherwise, uses Synthetic dataset for safe dry-run / pipeline testing.
    """
    train_transform, val_transform = get_armario_transforms(img_size)

    # Check if folder has valid class subdirectories with images
    has_image_folders = (
        os.path.exists(data_dir) and
        any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir) if not d.startswith('.'))
    )

    if has_image_folders:
        try:
            full_dataset = ImageFolder(root=data_dir, transform=train_transform)
            detected_classes = len(full_dataset.classes)
            logger.info(f"[DATASET] Loaded ImageFolder with {len(full_dataset)} images across {detected_classes} classes.")

            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            val_dataset.dataset.transform = val_transform
            n_classes = detected_classes
        except Exception as e:
            logger.warning(f"[AVISO] No se pudieron cargar imagenes desde {data_dir} ({e}).")
            logger.warning("[INFO] Generando lote de simulacion sintetica para verificar arquitectura y entrenamiento...")
            train_dataset = SyntheticArmarioDataset(320, num_classes, img_size)
            val_dataset = SyntheticArmarioDataset(80, num_classes, img_size)
            n_classes = num_classes
    else:
        logger.warning(f"[AVISO] No se encontraron carpetas de imagenes en '{data_dir}'.")
        logger.warning("[INFO] Para entrenar con DeepFashion2 real, proporciona los enlaces (--link_train) o sube el dataset.")
        logger.warning("[INFO] Iniciando con lote de prueba de 400 muestras sinteticas para verificar arquitectura, metricas y guardado en Drive...")
        train_dataset = SyntheticArmarioDataset(320, num_classes, img_size)
        val_dataset = SyntheticArmarioDataset(80, num_classes, img_size)
        n_classes = num_classes

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

    return train_loader, val_loader, n_classes
