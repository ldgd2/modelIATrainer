import os
import subprocess
import zipfile
from typing import Optional
from common.logger import setup_logger
from common.kaggle_utils import descargar_con_kagglehub, descargar_dataset_kaggle

logger = setup_logger("Diagnostico-Downloader")

SLUGS_DATASETS = {
    "colorimetria": "aibloy/fairface",
    "morfologia": "asad11914/coco-2017-keypoints",
    "capilar": "ipythonx/celebamaskhq"
}

def ejecutar_comando(comando: str):
    subprocess.run(comando, shell=True, check=True)

def descargar_directo(url: str, dest_dir: str, nombre_zip: str = "dataset.zip") -> str:
    """Descarga un archivo zip desde una URL y lo descomprime."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, nombre_zip)

    print(f"[*] Descargando desde: {url}...")
    try:
        ejecutar_comando(f'wget -q -O "{zip_path}" "{url}"')
    except Exception:
        ejecutar_comando(f'curl -s -L -o "{zip_path}" "{url}"')

    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        print(f"[*] Descomprimiendo en {dest_dir}...")
        try:
            ejecutar_comando(f'unzip -q "{zip_path}" -d "{dest_dir}"')
        except Exception:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        try:
            os.remove(zip_path)
        except Exception:
            pass
        print("[*] Archivo descomprimido exitosamente.")

    return dest_dir

def ensure_diagnostico_dataset(task: str, temp_path: str, direct_url: str = "") -> str:
    """
    Descarga y prepara el dataset correspondiente para la tarea de diagnostico morfologico.
    """
    dest_dir = os.path.join(temp_path, "dataset", task)
    os.makedirs(dest_dir, exist_ok=True)

    if any(os.scandir(dest_dir)):
        logger.info(f"[DATASET] Dataset para {task} ya presente en: {dest_dir}")
        return dest_dir

    if direct_url:
        return descargar_directo(direct_url, dest_dir, f"{task}.zip")

    slug = SLUGS_DATASETS.get(task, "aibloy/fairface")
    logger.info(f"[*] Descargando dataset oficial para {task} ({slug}) via kagglehub...")
    path = descargar_con_kagglehub(slug)
    if path and os.path.exists(path):
        logger.info(f"[SUCCESS] Dataset listo en: {path}")
        return path

    return dest_dir
