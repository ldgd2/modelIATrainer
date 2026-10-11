import os
from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
from models.armario_digital.dataset.transforms import get_armario_transforms
from common.logger import setup_logger

logger = setup_logger("Armario-Dataset")

class DeepFashionCategoryDataset(Dataset):
    """Dataset para cargar las anotaciones oficiales de DeepFashion (50 categorias reales)."""
    def __init__(self, anno_path: str, img_root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(anno_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        start_idx = 0
        if len(lines) > 2 and lines[0].isdigit():
            start_idx = 2

        # Detectar prefijo base donde estan las imagenes fisicas
        first_rel = lines[start_idx].split()[0]
        base_dir = img_root
        for prefix in ["", "Img", "img", "Img/img", "versions/1"]:
            cand = os.path.join(img_root, prefix, first_rel)
            if os.path.exists(cand):
                base_dir = os.path.join(img_root, prefix)
                break

        max_cat = 0
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 2:
                rel_path = parts[0]
                try:
                    cat_id = int(parts[1]) - 1  # 1-indexed -> 0-indexed
                    if cat_id < 0:
                        cat_id = 0
                    full_p = os.path.join(base_dir, rel_path)
                    self.samples.append((full_p, cat_id))
                    if cat_id > max_cat:
                        max_cat = cat_id
                except ValueError:
                    continue

        self.num_classes = max_cat + 1
        logger.info(f"[DATASET] Anotaciones DeepFashion cargadas: {len(self.samples)} imagenes en {self.num_classes} categorias.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback en caso de imagen corrupta
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label

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
    Soporta:
    1. Anotaciones oficiales DeepFashion (list_category_img.txt con 50 clases).
    2. Carpetas estructuradas de clases (ImageFolder).
    3. Descubrimiento automatico de subcarpetas anidadas.
    """
    train_transform, val_transform = get_armario_transforms(img_size)

    # 1. Buscar si existe archivo de anotaciones de categorias de DeepFashion
    anno_file = None
    anno_candidates = [
        os.path.join(data_dir, "Anno", "list_category_img.txt"),
        os.path.join(data_dir, "list_category_img.txt"),
        os.path.join(os.path.dirname(data_dir), "Anno", "list_category_img.txt"),
        os.path.join(os.path.dirname(data_dir), "list_category_img.txt"),
    ]
    for cand in anno_candidates:
        if os.path.exists(cand):
            anno_file = cand
            break

    if anno_file:
        logger.info(f"[*] Anotaciones encontradas en: {anno_file}")
        try:
            full_dataset = DeepFashionCategoryDataset(anno_file, data_dir, transform=train_transform)
            if len(full_dataset) > 0:
                val_size = int(len(full_dataset) * val_split)
                train_size = len(full_dataset) - val_size
                train_dataset, val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                val_dataset.dataset.transform = val_transform
                n_classes = full_dataset.num_classes
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                return train_loader, val_loader, n_classes
        except Exception as e:
            logger.warning(f"No se pudo cargar DeepFashionCategoryDataset ({e}). Probando ImageFolder...")

    # 2. Desenrollar subcarpetas intermedias (ej. img, Img, images)
    if os.path.exists(data_dir):
        for sub in ["img", "Img", "images", "clothing-dataset-full", "train"]:
            candidate = os.path.join(data_dir, sub)
            if os.path.exists(candidate) and os.path.isdir(candidate):
                subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d)) and not d.startswith('.')]
                if len(subdirs) > 1:
                    data_dir = candidate
                    break

    has_image_folders = (
        os.path.exists(data_dir) and
        any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir) if not d.startswith('.'))
    )

    if has_image_folders:
        try:
            full_dataset = ImageFolder(root=data_dir, transform=train_transform)
            detected_classes = len(full_dataset.classes)

            # Si detecta 1 sola clase, verificar si esa clase es en realidad un directorio que contiene las categorias
            if detected_classes == 1:
                single_folder = os.path.join(data_dir, full_dataset.classes[0])
                if os.path.isdir(single_folder):
                    inner_subdirs = [d for d in os.listdir(single_folder) if os.path.isdir(os.path.join(single_folder, d)) and not d.startswith('.')]
                    if len(inner_subdirs) > 1:
                        logger.info(f"[*] Subcarpetas de categorias detectadas dentro de '{full_dataset.classes[0]}': {len(inner_subdirs)} clases.")
                        data_dir = single_folder
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
            train_dataset = SyntheticArmarioDataset(320, num_classes, img_size)
            val_dataset = SyntheticArmarioDataset(80, num_classes, img_size)
            n_classes = num_classes
    else:
        logger.warning(f"[AVISO] No se encontraron carpetas de imagenes en '{data_dir}'.")
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
