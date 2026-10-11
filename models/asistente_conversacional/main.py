import os
import sys
import argparse
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager, find_latest_checkpoint
from models.asistente_conversacional.architecture.network import build_llm_model
from models.asistente_conversacional.dataset.dataset import build_llm_dataloaders
from models.asistente_conversacional.trainer.trainer import LLMTrainer
from models.asistente_conversacional.download.download_llm import descargar_modelo_gguf
from models.asistente_conversacional.trainer.benchmark import evaluar_modelo_local

logger = setup_logger("AsistenteConversacional-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/asistente_conversacional"
DEFAULT_TEMP_DIR = "/content/temp_asistente"
DEFAULT_FILENAME = "bellai_assistant_8b.gguf"
DEFAULT_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Asistente Conversacional (Fine-Tuning Estilista y Descarga GGUF)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "download"], help="Modo: 'train' (Fine-Tuning) o 'download'")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Modelo base Hugging Face para Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Numero de epocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--custom_dataset", type=str, default="", help="Ruta opcional a dataset JSON/JSONL de estilismo")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO, help="Repositorio Hugging Face para modo download")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Nombre de archivo .gguf para modo download")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--resume", type=str, default="", help="Ruta a checkpoint o 'auto'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"[*] Iniciando modulo Asistente Conversacional (Modo: {args.mode})...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    if args.mode == "download":
        dest_file = descargar_modelo_gguf(repo_id=args.repo_id, filename=args.filename, dest_dir=args.drive_path)
        if dest_file:
            evaluar_modelo_local(dest_file)
            logger.info(f"Modelo GGUF listo en: {dest_file}")
        return

    # MODO TRAIN (Fine-Tuning de Estilista BellAI)
    logger.info(f"[*] Configurando Fine-Tuning de Asistente Estilista BellAI sobre {args.base_model}...")

    # 1. Cargar modelo base y tokenizer
    model, tokenizer = build_llm_model(base_model_name=args.base_model, use_lora=True)

    # 2. Cargar DataLoader de instrucciones de estilismo
    train_loader = build_llm_dataloaders(
        data_path=args.custom_dataset if args.custom_dataset else None,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    # 3. Checkpoint y recuperacion automatica de epoca
    start_epoch = 1
    resume_path = args.resume

    if resume_path == "auto" or not resume_path:
        found_ckpt, found_epoch = find_latest_checkpoint(args.drive_path)
        if found_ckpt:
            resume_path = found_ckpt
            start_epoch = found_epoch + 1
            logger.info(f"[*] Checkpoint detectado automaticamente en Drive: {found_ckpt}")
            logger.info(f"[*] Continuando fine-tuning desde la epoca {start_epoch} (completadas: {found_epoch})")
    elif resume_path and os.path.exists(resume_path):
        import re
        match = re.search(r"epoch_(\d+)", resume_path)
        if match:
            start_epoch = int(match.group(1)) + 1
            logger.info(f"[*] Checkpoint especificado: {resume_path} -> Continuando desde la epoca {start_epoch}")

    if resume_path and os.path.exists(resume_path):
        try:
            device = torch.device(args.device)
            state = torch.load(resume_path, map_location=device)
            weights = state.get("model_state", state)
            model.load_state_dict(weights, strict=False)
            logger.info(f"Pesos restaurados desde checkpoint: {resume_path}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    # 4. Control acumulativo de epocas
    total_epochs = args.epochs
    if start_epoch > total_epochs:
        logger.info(f"[*] El checkpoint ya alcanzo la epoca {start_epoch - 1}. Sumando {args.epochs} epocas adicionales al entrenamiento.")
        total_epochs = (start_epoch - 1) + args.epochs
        logger.info(f"[*] Nueva meta de entrenamiento: Epoca {start_epoch} hasta {total_epochs}")

    # 5. Configuracion del entrenador
    config = {
        "epochs": total_epochs,
        "start_epoch": start_epoch,
        "learning_rate": args.learning_rate,
        "output_dir": os.path.join(args.drive_path, "checkpoints"),
        "drive_backup_dir": args.drive_path,
        "device": args.device
    }

    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        tokenizer=tokenizer
    )

    # 6. Ejecutar entrenamiento
    trainer.fit()

    # 7. Release final
    storage.save_release(model, "bellai_stylist_v1")
    logger.info("[SUCCESS] Asistente Conversacional BellAI fine-tuneado y respaldado en Drive.")

if __name__ == "__main__":
    main()
