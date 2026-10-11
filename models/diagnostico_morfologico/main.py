import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from common.kaggle_utils import descargar_con_kagglehub

logger = setup_logger("DiagnosticoMorfologico-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/diagnostico_morfologico"
DEFAULT_TEMP_DIR = "/content/temp_diagnostico"

def build_model(task: str):
    if task == "colorimetria":
        # Clasificador de tonos/estaciones (FairFace)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
    elif task == "morfologia":
        # Estimador de proporciones corporales y keypoints (COCO Keypoints)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 34)
    elif task == "capilar":
        # Segmentador capilar (CelebAMask-HQ)
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Diagnostico Morfologico")
    parser.add_argument("--task", type=str, default="colorimetria", choices=["colorimetria", "morfologia", "capilar"], help="Sub-tarea a entrenar")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--download_data", action="store_true", help="Descargar dataset con kagglehub")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint a restaurar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Iniciando Diagnostico Morfologico (Tarea: {args.task})...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    # Descarga via kagglehub segun la tarea
    slug_map = {
        "colorimetria": "aibloy/fairface",
        "morfologia": "asad11914/coco-2017-keypoints",
        "capilar": "ipythonx/celebamaskhq"
    }
    release_map = {
        "colorimetria": "colorimetry_v1",
        "morfologia": "morphology_v1",
        "capilar": "hair_analysis_v1"
    }

    if args.download_data:
        slug = slug_map[args.task]
        data_path = descargar_con_kagglehub(slug)
        logger.info(f"Dataset disponible en: {data_path}")

    device = torch.device(args.device)
    model = build_model(args.task).to(device)

    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            logger.info(f"Pesos restaurados desde: {args.resume}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() if args.task in ["morfologia", "colorimetria"] else nn.BCELoss()

    logger.info(f"Iniciando entrenamiento por {args.epochs} epocas en {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        dummy_in = torch.randn(args.batch_size, 3, 224, 224 if args.task != "capilar" else 64).to(device)
        dummy_target = torch.randn(args.batch_size, 4 if args.task == "colorimetria" else (34 if args.task == "morfologia" else 1)).to(device)
        if args.task == "capilar":
            dummy_target = torch.randint(0, 2, (args.batch_size, 1, 64, 64)).float().to(device)

        optimizer.zero_grad()
        out = model(dummy_in)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer.step()

        logger.info(f"Epoca [{epoch:02d}/{args.epochs:02d}] | Loss: {loss.item():.4f}")

        if epoch % args.save_every == 0:
            ckpt_name = f"{args.task}_epoch_{epoch}.pt"
            storage.save_checkpoint(model, ckpt_name)

    release_name = release_map[args.task]
    storage.save_release(model, release_name)

if __name__ == "__main__":
    main()
