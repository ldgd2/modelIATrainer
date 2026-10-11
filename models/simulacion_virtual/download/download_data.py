import os
from common.logger import setup_logger
from common.kaggle_utils import descargar_con_kagglehub

logger = setup_logger("Simulacion-Downloader")

SLUGS_SIMULACION = {
    "probador": "marquis03/high-resolution-viton-zalando-dataset",
    "maquillaje": "cyd0806/makeup-transfer-dataset"
}

def ensure_simulacion_dataset(task: str, temp_path: str) -> str:
    """
    Descarga y prepara los datasets para Probador Virtual (VITON-HD)
    o Simulacion de Maquillaje (MT-Dataset) mediante kagglehub.
    """
    dest_dir = os.path.join(temp_path, "dataset", task)
    os.makedirs(dest_dir, exist_ok=True)

    if any(os.scandir(dest_dir)):
        logger.info(f"[DATASET] Dataset para {task} ya presente en: {dest_dir}")
        return dest_dir

    slug = SLUGS_SIMULACION.get(task, SLUGS_SIMULACION["probador"])
    logger.info(f"[*] Descargando dataset oficial para {task} ({slug}) via kagglehub...")
    path = descargar_con_kagglehub(slug)
    if path and os.path.exists(path):
        logger.info(f"[SUCCESS] Dataset listo en: {path}")
        return path

    return dest_dir
