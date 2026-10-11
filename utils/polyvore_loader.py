import os
import json
import random
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PolyVoreCompatibilityDataset(Dataset):
    """
    Dataset para pares de prendas (compatibles e incompatibles) de Re-PolyVore.
    """
    def __init__(self, data_dir: str, transform=None, num_pairs: int = 1000):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []

        # Buscar recursivamente imagenes jpg/png en la carpeta descomprimida
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_files.append(os.path.join(root, f))

        self.num_pairs = min(num_pairs, len(self.image_files) // 2) if len(self.image_files) >= 2 else 0

    def __len__(self) -> int:
        return max(self.num_pairs, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Para pares positivos (mismo contexto) y negativos (muestreo aleatorio)
        es_compatible = 1.0 if (idx % 2 == 0) else 0.0

        img_path_a = self.image_files[idx % len(self.image_files)]
        if es_compatible == 1.0:
            img_path_b = self.image_files[(idx + 1) % len(self.image_files)]
        else:
            rand_idx = random.randint(0, len(self.image_files) - 1)
            img_path_b = self.image_files[rand_idx]

        try:
            img_a = Image.open(img_path_a).convert("RGB")
            img_b = Image.open(img_path_b).convert("RGB")
        except Exception:
            img_a = Image.new("RGB", (224, 224), color=(128, 128, 128))
            img_b = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        target = torch.tensor(es_compatible, dtype=torch.float32)
        return img_a, img_b, target

class SyntheticPolyVoreDataset(Dataset):
    """Dataset sintetico para validacion del pipeline sin datos descargados."""
    def __init__(self, size: int = 64, img_size: int = 224):
        self.size = size
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_a = torch.rand(3, self.img_size, self.img_size)
        img_b = torch.rand(3, self.img_size, self.img_size)
        label = torch.tensor(1.0 if idx % 2 == 0 else 0.0, dtype=torch.float32)
        return img_a, img_b, label

def obtener_polyvore_transforms(img_size: int = 224):
    """Transformaciones estandar para imagenes de PolyVore."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def obtener_polyvore_loaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Instancia train_loader y val_loader para compatibilidad PolyVore."""
    tf = obtener_polyvore_transforms(img_size)

    ds = PolyVoreCompatibilityDataset(data_dir, transform=tf)
    if len(ds) >= 10:
        print(f"[*] Dataset PolyVore cargado con {len(ds)} pares de entrenamiento.")
        train_size = int(len(ds) * 0.8)
        val_size = len(ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    else:
        print("[INFO] Usando dataset sintetico de prueba para PolyVore.")
        train_ds = SyntheticPolyVoreDataset(size=64, img_size=img_size)
        val_ds = SyntheticPolyVoreDataset(size=32, img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
