import os
import sys
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from models.motor_recomendacion.download.download_data import descargar_y_extraer_polyvore
from models.motor_recomendacion.dataset.dataset import build_recommender_dataloaders
from models.motor_recomendacion.architecture.network import StyleRecommenderNet
from models.motor_recomendacion.trainer.trainer import RecommenderTrainer

logger = setup_logger("MotorRecomendacion-Main")

DEFAULT_POLYVORE_URL = (
    "https://hkaidlab-my.sharepoint.com/personal/xingxingzou_aidlab_hk/_layouts/15/download.aspx"
    "?SourceUrl=%2Fpersonal%2Fxingxingzou_aidlab_hk%2FDocuments%2FAiDLab%20-%20fAshIon%20TEAM%20-%20DATA%2FRe-PolyVore.zip"
)
DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/motor_recomendacion"
DEFAULT_TEMP_DIR = "/content/temp_recomendacion"

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Motor de Recomendacion (PolyVore -> style_recommender_v1.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension de embedding")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--link_polyvore", type=str, default=DEFAULT_POLYVORE_URL, help="URL de Re-PolyVore.zip")
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint a restaurar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("[*] Iniciando Motor de Recomendacion de Outfits (PolyVore)...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    # 1. Descarga o extraccion de datos
    data_dir = descargar_y_extraer_polyvore(args.link_polyvore, args.temp_path)

    # 2. Carga de DataLoaders
    train_loader, val_loader = build_recommender_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size
    )

    # 3. Modelo
    model = StyleRecommenderNet(embedding_dim=args.embedding_dim)

    # 4. Checkpoint
    if args.resume and os.path.exists(args.resume):
        try:
            device = torch.device(args.device)
            state = torch.load(args.resume, map_location=device)
            weights = state.get("model_state", state)
            model.load_state_dict(weights)
            logger.info(f"Pesos restaurados desde: {args.resume}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    # 5. Entrenador
    config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "output_dir": os.path.join(args.drive_path, "checkpoints"),
        "drive_backup_dir": args.drive_path,
        "device": args.device
    }

    trainer = RecommenderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 6. Ejecucion
    trainer.fit()

    # 7. Release
    storage.save_release(model, "style_recommender_v1")

if __name__ == "__main__":
    main()
