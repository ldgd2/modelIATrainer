import os
import sys
import argparse
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from common.kaggle_utils import descargar_con_kagglehub

logger = setup_logger("SimulacionVirtual-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/simulacion_virtual"
DEFAULT_TEMP_DIR = "/content/temp_simulacion"

def build_model():
    return nn.Sequential(
        nn.Conv2d(6, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 3, kernel_size=3, padding=1),
        nn.Tanh()
    )

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Simulacion Virtual")
    parser.add_argument("--task", type=str, default="probador", choices=["probador", "maquillaje"], help="Sub-tarea a entrenar")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--download_data", action="store_true", help="Descargar dataset con kagglehub")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint a restaurar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Iniciando Simulacion Virtual (Tarea: {args.task})...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    slug_map = {
        "probador": "marquis03/high-resolution-viton-zalando-dataset",
        "maquillaje": "cyd0806/makeup-transfer-dataset"
    }
    release_map = {
        "probador": "virtual_tryon_v1",
        "maquillaje": "makeup_sim_v1"
    }

    if args.download_data:
        slug = slug_map[args.task]
        data_path = descargar_con_kagglehub(slug)
        logger.info(f"Dataset disponible en: {data_path}")

    device = torch.device(args.device)
    model = build_model().to(device)

    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            logger.info(f"Pesos restaurados desde: {args.resume}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    from tqdm.auto import tqdm
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()
    batches_per_epoch = 20

    logger.info(f"Iniciando entrenamiento por {args.epochs} epocas en {device} ({batches_per_epoch} batches por epoca)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoca {epoch:02d}/{args.epochs:02d} [{args.task}]", leave=True, ncols=100)

        for _ in pbar:
            dummy_in = torch.randn(args.batch_size, 6, 128, 128).to(device)
            dummy_target = torch.randn(args.batch_size, 3, 128, 128).to(device)

            optimizer.zero_grad()
            out = model(dummy_in)
            loss = criterion(out, dummy_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / batches_per_epoch
        logger.info(f"Epoca [{epoch:02d}/{args.epochs:02d}] completada | Loss promedio: {avg_loss:.4f}")

        if epoch % args.save_every == 0:
            ckpt_name = f"{args.task}_epoch_{epoch}.pt"
            storage.save_checkpoint(model, ckpt_name)

    release_name = release_map[args.task]
    storage.save_release(model, release_name)

if __name__ == "__main__":
    main()
