import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from models.asistente_conversacional.download.download_llm import descargar_modelo_gguf

logger = setup_logger("AsistenteConversacional-Main")

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/asistente_conversacional"
DEFAULT_FILENAME = "bellai_assistant_8b.gguf"
DEFAULT_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Asistente Conversacional (LLM Local GGUF)")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO, help="Repositorio Hugging Face")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Nombre de archivo .gguf")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default="/content/temp_asistente", help="Ruta temporal local")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Iniciando modulo Asistente Conversacional (LLM Local)...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    descargar_modelo_gguf(repo_id=args.repo_id, filename=args.filename, dest_dir=args.drive_path)
    logger.info(f"Modelo GGUF listo en: {os.path.join(args.drive_path, args.filename)}")

if __name__ == "__main__":
    main()
