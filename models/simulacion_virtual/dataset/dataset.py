import os
import random
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from common.logger import setup_logger

logger = setup_logger("SimulacionVirtual-Dataset")

class RealSimulacionDataset(Dataset):
    """Carga pares de imagenes reales para probador virtual o maquillaje."""
    def __init__(self, image_paths: List[str], img_size: int = 128, task: str = "probador", transform=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.task = task
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_a = self.image_paths[idx]
        other_idx = (idx + random.randint(1, 5)) % len(self.image_paths)
        path_b = self.image_paths[other_idx]

        try:
            img_a = Image.open(path_a).convert("RGB")
        except Exception:
            img_a = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        try:
            img_b = Image.open(path_b).convert("RGB")
        except Exception:
            img_b = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        t_a = self.transform(img_a)
        t_b = self.transform(img_b)

        # Entrada 6 canales: persona + prenda
        img_in = torch.cat([t_a, t_b], dim=0)
        # Salida 3 canales: objetivo de sintesis
        img_target = t_a
        return img_in, img_target

class SyntheticSimulacionDataset(Dataset):
    """Generador sintetico de pares persona-prenda o rostro-referencia para prueba de pipeline."""
    def __init__(self, num_samples: int = 320, img_size: int = 128, task: str = "probador"):
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
    batch_size: int = 16,
    img_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"[*] Construyendo DataLoaders para {task} en: {data_dir}")

    # Buscar imagenes reales en disco
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    found_images = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    found_images.append(os.path.join(root, f))
            if len(found_images) >= 30000:
                break

    if len(found_images) >= 20:
        logger.info(f"[DATASET] {len(found_images)} imagenes reales para {task} encontradas en {data_dir}.")
        random.shuffle(found_images)
        val_count = max(int(len(found_images) * 0.15), 10)
        train_images = found_images[val_count:]
        val_images = found_images[:val_count]
        train_dataset = RealSimulacionDataset(train_images, img_size=img_size, task=task)
        val_dataset = RealSimulacionDataset(val_images, img_size=img_size, task=task)
    else:
        logger.warning(f"[AVISO] Directorio '{data_dir}' sin imagenes extraidas. Usando lote estructurado (320 muestras)...")
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
