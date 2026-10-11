import os
import random
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from common.logger import setup_logger

logger = setup_logger("MotorRecomendacion-Dataset")

class RealPolyvoreDataset(Dataset):
    """Carga pares reales de prendas desde el dataset PolyVore en disco."""
    def __init__(self, image_paths: List[str], img_size: int = 224, transform=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_a = self.image_paths[idx]
        is_pos = (random.random() > 0.5)
        if is_pos:
            other_idx = (idx + random.randint(1, 5)) % len(self.image_paths)
            target = 1.0
        else:
            other_idx = (idx + random.randint(10, max(len(self.image_paths) - 1, 11))) % len(self.image_paths)
            target = 0.0

        path_b = self.image_paths[other_idx]
        try:
            img_a = Image.open(path_a).convert("RGB")
        except Exception:
            img_a = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        try:
            img_b = Image.open(path_b).convert("RGB")
        except Exception:
            img_b = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        return self.transform(img_a), self.transform(img_b), torch.tensor(target, dtype=torch.float32)

class SyntheticPolyvoreDataset(Dataset):
    """Generador de pares sintetico para verificacion del pipeline y prueba funcional."""
    def __init__(self, num_samples: int = 400, img_size: int = 224):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item_a = torch.rand(3, self.img_size, self.img_size)
        item_b = torch.rand(3, self.img_size, self.img_size)
        target = float(idx % 2 == 0)
        return item_a, item_b, torch.tensor(target, dtype=torch.float32)

def build_recommender_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"[*] Construyendo DataLoaders para motor de recomendacion en: {data_dir}")

    # Buscar imagenes reales en disco
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    found_images = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    found_images.append(os.path.join(root, f))
            if len(found_images) >= 50000:
                break

    if len(found_images) >= 20:
        logger.info(f"[DATASET] {len(found_images)} imagenes reales de PolyVore encontradas en {data_dir}.")
        random.shuffle(found_images)
        val_count = max(int(len(found_images) * 0.15), 10)
        train_images = found_images[val_count:]
        val_images = found_images[:val_count]
        train_dataset = RealPolyvoreDataset(train_images, img_size=img_size)
        val_dataset = RealPolyvoreDataset(val_images, img_size=img_size)
    else:
        logger.warning(f"[AVISO] Carpeta '{data_dir}' sin imagenes extraidas. Utilizando lote de prueba estructurada (400 muestras)...")
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
