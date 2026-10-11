import os
import json
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DeepFashion2Dataset(Dataset):
    """
    Dataset adaptado para DeepFashion2.
    Estructura esperada:
      directorio/
        image/ (archivos .jpg)
        annos/ (archivos .json con category_id 1..13)
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        image_dir = os.path.join(root_dir, "image")
        annos_dir = os.path.join(root_dir, "annos")

        if os.path.exists(image_dir) and os.path.exists(annos_dir):
            for file_name in os.listdir(annos_dir):
                if file_name.endswith(".json"):
                    base_id = os.path.splitext(file_name)[0]
                    img_path = os.path.join(image_dir, f"{base_id}.jpg")
                    json_path = os.path.join(annos_dir, file_name)

                    if os.path.exists(img_path):
                        self.samples.append((img_path, json_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, json_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extraer categoria principal del item1 (1-indexed a 0-indexed)
            category_id = data.get("item1", {}).get("category_id", 1) - 1
            category_id = max(0, min(category_id, 12))
        except Exception:
            category_id = 0

        if self.transform:
            image = self.transform(image)

        return image, category_id

class FallbackDataset(Dataset):
    """Dataset sintetico para validacion rapida del flujo sin datos descargados."""
    def __init__(self, size: int = 64, num_classes: int = 13, img_size: int = 224):
        self.size = size
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = torch.rand(3, self.img_size, self.img_size)
        label = idx % self.num_classes
        return img, label

def obtener_transformaciones(img_size: int = 224):
    """Devuelve las transformaciones de entrenamiento y validacion."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def crear_dataset(data_dir: str, transform, is_train: bool = True) -> Dataset:
    """Detecta automaticamente el formato del dataset (DeepFashion2, carpetas de clase o fallback)."""
    if not os.path.exists(data_dir):
        print(f"[INFO] Directorio {data_dir} no existe. Usando dataset sintetico de prueba.")
        return FallbackDataset(size=64 if is_train else 32)

    image_subdir = os.path.join(data_dir, "image")
    annos_subdir = os.path.join(data_dir, "annos")

    # 1. DeepFashion2 format
    if os.path.exists(image_subdir) and os.path.exists(annos_subdir):
        ds = DeepFashion2Dataset(data_dir, transform=transform)
        if len(ds) > 0:
            print(f"[INFO] Cargado DeepFashion2 desde {data_dir}: {len(ds)} muestras.")
            return ds

    # 2. ImageFolder format (subcarpetas por categoria)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs and not ('image' in subdirs and len(subdirs) == 1):
        try:
            ds = ImageFolder(root=data_dir, transform=transform)
            print(f"[INFO] Cargado ImageFolder desde {data_dir}: {len(ds)} imagenes.")
            return ds
        except Exception:
            pass

    print(f"[INFO] Directorio {data_dir} sin formato reconocido. Usando dataset sintetico.")
    return FallbackDataset(size=64 if is_train else 32)

def obtener_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Instancia y retorna train_loader y val_loader."""
    train_tf, val_tf = obtener_transformaciones(img_size)

    train_dataset = crear_dataset(train_dir, train_tf, is_train=True)
    val_dataset = crear_dataset(val_dir, val_tf, is_train=False)

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
