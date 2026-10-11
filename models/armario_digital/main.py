import os
import sys
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from models.armario_digital.download.download_data import preparar_dataset, ensure_dataset
from models.armario_digital.dataset.dataset import build_dataloaders
from models.armario_digital.architecture.network import build_model
from models.armario_digital.trainer.trainer import ArmarioTrainer

logger = setup_logger("ArmarioDigital-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/armario_digital"
DEFAULT_TEMP_DIR = "/content/temp_armario"

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Armario Digital (DeepFashion2 -> wardrobe_v1.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=13, help="Categorias de prendas")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers DataLoader")
    parser.add_argument("--password", type=str, default="2019Deepfashion2**", help="Password descompresion")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Directorio en Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Directorio temporal local")

    # Enlaces de descarga directa
    parser.add_argument("--link_train", type=str, default="", help="Enlace wget para train.zip")
    parser.add_argument("--link_val", type=str, default="", help="Enlace wget para val.zip")
    parser.add_argument("--link_test", type=str, default="", help="Enlace wget para test.zip")

    # Rutas locales y restauracion
    parser.add_argument("--dataset", type=str, default="", help="Ruta local si ya existe dataset")
    parser.add_argument("--resume", type=str, default="", help="Ruta a checkpoint para continuar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Iniciando modulo Armario Digital...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    links = {"train": args.link_train, "val": args.link_val, "test": args.link_test}
    if any(links.values()):
        dataset_paths = preparar_dataset(links, args.temp_path, password=args.password)
        data_dir = dataset_paths.get("train", os.path.join(args.temp_path, "dataset", "train"))
    elif args.dataset:
        data_dir = args.dataset
    else:
        data_dir = ensure_dataset(os.path.join(args.temp_path, "dataset", "train"))

    train_loader, val_loader, detected_classes = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes
    )

    device = torch.device(args.device)
    model = build_model(num_classes=detected_classes, pretrained=True).to(device)

    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            logger.info(f"Pesos restaurados desde: {args.resume}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "output_dir": os.path.join(args.drive_path, "checkpoints"),
        "drive_backup_dir": args.drive_path,
        "device": args.device
    }

    trainer = ArmarioTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    trainer.fit()

    storage.save_release(model, "wardrobe_v1")

if __name__ == "__main__":
    main()
