import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from common.logger import setup_logger

logger = setup_logger("Diagnostico-Dataset")

class SyntheticDiagnosticoDataset(Dataset):
    """Dataset sintetico de alta velocidad para verificacion de pipeline y ejecucion rapida."""
    def __init__(self, num_samples: int = 400, task: str = "colorimetria", img_size: int = 224):
        self.num_samples = num_samples
        self.task = task
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        c = 3
        h = self.img_size
        w = self.img_size
        img = torch.rand(c, h, w)

        if self.task == "colorimetria":
            # 4 estaciones de color (primavera, verano, otono, invierno)
            target = idx % 4
        elif self.task == "morfologia":
            # 34 valores de keypoints / proporciones corporales
            target = torch.randn(34)
        elif self.task == "capilar":
            # Mascara binaria de segmentacion capilar (1, 64, 64)
            target = torch.randint(0, 2, (1, 64, 64)).float()
        else:
            target = idx % 2

        return img, target

def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def build_dataloaders(
    data_dir: str,
    task: str = "colorimetria",
    batch_size: int = 16,
    img_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    train_transform, val_transform = get_transforms(img_size)

    has_folders = (
        os.path.exists(data_dir) and
        any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir) if not d.startswith('.'))
    )

    if has_folders and task == "colorimetria":
        try:
            full_dataset = ImageFolder(root=data_dir, transform=train_transform)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            logger.info(f"[DATASET] Cargadas {len(full_dataset)} imagenes desde {data_dir}.")
        except Exception as e:
            logger.warning(f"[AVISO] Fallo al cargar ImageFolder ({e}). Usando lote de simulacion...")
            train_dataset = SyntheticDiagnosticoDataset(320, task, img_size)
            val_dataset = SyntheticDiagnosticoDataset(80, task, img_size)
    else:
        logger.info(f"[INFO] Preparando lote estructurado para {task} (400 muestras para verificacion de entrenamiento)...")
        train_dataset = SyntheticDiagnosticoDataset(320, task, img_size)
        val_dataset = SyntheticDiagnosticoDataset(80, task, img_size)

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
