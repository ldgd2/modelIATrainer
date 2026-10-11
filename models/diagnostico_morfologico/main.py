import os
import sys
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from models.diagnostico_morfologico.download.download_data import ensure_diagnostico_dataset
from models.diagnostico_morfologico.dataset.dataset import build_dataloaders
from models.diagnostico_morfologico.architecture.network import build_diagnostico_model
from models.diagnostico_morfologico.trainer.trainer import DiagnosticoTrainer

logger = setup_logger("DiagnosticoMorfologico-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/diagnostico_morfologico"
DEFAULT_TEMP_DIR = "/content/temp_diagnostico"

RELEASE_NAMES = {
    "colorimetria": "colorimetry_v1",
    "morfologia": "morphology_v1",
    "capilar": "hair_analysis_v1"
}

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Diagnostico Morfologico (FairFace, COCO Keypoints, CelebAMask-HQ)")
    parser.add_argument("--task", type=str, default="colorimetria", choices=["colorimetria", "morfologia", "capilar"], help="Sub-tarea a entrenar")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--download_data", action="store_true", help="Descargar dataset correspondiente con kagglehub")
    parser.add_argument("--direct_url", type=str, default="", help="Enlace de descarga directa opcional")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint a restaurar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"[*] Iniciando Diagnostico Morfologico (Tarea: {args.task})...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    # 1. Gestion de datos (Descarga o preparacion)
    data_dir = os.path.join(args.temp_path, "dataset", args.task)
    if args.download_data or args.direct_url:
        data_dir = ensure_diagnostico_dataset(args.task, args.temp_path, direct_url=args.direct_url)

    # 2. Construccion de DataLoaders
    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        task=args.task,
        batch_size=args.batch_size
    )

    # 3. Construccion de arquitectura de red
    model = build_diagnostico_model(args.task)

    # 4. Restauracion y autocomprobacion de checkpoints
    start_epoch = 1
    resume_path = args.resume.strip() if args.resume else ""

    if resume_path == "auto" or (resume_path == "" and os.path.exists(os.path.join(args.drive_path, "checkpoints"))):
        from common.storage import find_latest_checkpoint
        auto_ckpt, last_ep = find_latest_checkpoint(args.drive_path)
        if auto_ckpt:
            resume_path = auto_ckpt
            logger.info(f"[*] Autodetectado ultimo checkpoint en Drive: {auto_ckpt} (Epoca completada: {last_ep})")

    if resume_path and os.path.exists(resume_path):
        try:
            device = torch.device(args.device)
            ckpt_data = torch.load(resume_path, map_location=device)
            weights = ckpt_data.get("model_state", ckpt_data)
            model.load_state_dict(weights)
            if isinstance(ckpt_data, dict) and "epoch" in ckpt_data:
                start_epoch = int(ckpt_data["epoch"]) + 1
            logger.info(f"[*] Pesos restaurados desde: {resume_path} | Se reanudara a partir de la epoca {start_epoch}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    # 5. Configuracion del entrenador
    config = {
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "output_dir": os.path.join(args.drive_path, "checkpoints"),
        "drive_backup_dir": args.drive_path,
        "device": args.device
    }

    trainer = DiagnosticoTrainer(
        model=model,
        task=args.task,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 6. Ejecucion del entrenamiento
    trainer.fit()

    # 7. Guardado de version release para produccion
    release_name = RELEASE_NAMES.get(args.task, f"{args.task}_v1")
    storage.save_release(model, release_name)

if __name__ == "__main__":
    main()
