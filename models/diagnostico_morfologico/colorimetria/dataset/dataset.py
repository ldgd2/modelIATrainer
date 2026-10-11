import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from models.colorimetria.dataset.transforms import get_colorimetria_transforms
from common.logger import setup_logger

logger = setup_logger("Colorimetria-Dataset")

class SyntheticColorimetriaDataset(Dataset):
    """Fallback dataset for testing facial colorimetry pipeline integrity."""
    def __init__(self, num_samples: int = 40, num_classes: int = 4, img_size: int = 224, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size)
        label = idx % self.num_classes
        return img, label

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.15,
    num_workers: int = 2,
    num_classes: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Creates train and validation DataLoaders for facial colorimetry dataset.
    """
    train_transform, val_transform = get_colorimetria_transforms(img_size)

    has_image_folders = (
        os.path.exists(data_dir) and
        any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir) if not d.startswith('.'))
    )

    if has_image_folders:
        try:
            full_dataset = ImageFolder(root=data_dir, transform=train_transform)
            detected_classes = len(full_dataset.classes)
            logger.info(f"[DATASET] Loaded ImageFolder with {len(full_dataset)} images across {detected_classes} palettes.")

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
            logger.warning(f"[WARNING] Failed to load images from {data_dir}: {e}. Using dummy dataset.")
            train_dataset = SyntheticColorimetriaDataset(60, num_classes, img_size)
            val_dataset = SyntheticColorimetriaDataset(20, num_classes, img_size)
            n_classes = num_classes
    else:
        logger.warning(f"[WARNING] No dataset folders found in '{data_dir}'. Using synthetic data for verification.")
        train_dataset = SyntheticColorimetriaDataset(60, num_classes, img_size)
        val_dataset = SyntheticColorimetriaDataset(20, num_classes, img_size)
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
