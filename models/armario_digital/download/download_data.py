import os
import subprocess
import zipfile
from typing import Dict, Optional
from common.logger import setup_logger

from common.download_utils import descargar_con_barra, descomprimir_con_barra

logger = setup_logger("Armario-Downloader")

def preparar_dataset(links: Dict[str, str], temp_path: str, password: Optional[str] = "2019Deepfashion2**") -> Dict[str, str]:
    """
    Descarga y descomprime los datasets en el almacenamiento efimero mostrando
    barra de progreso interactiva (porcentaje, MB descargados/totales y tiempo).
    """
    zips_dir = os.path.join(temp_path, "zips")
    data_dir = os.path.join(temp_path, "dataset")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    rutas_extraidas = {}

    for split_name, url in links.items():
        if url:
            zip_path = os.path.join(zips_dir, f"{split_name}.zip")
            extract_path = os.path.join(data_dir, split_name)

            # 1. Descarga interactiva con barra
            exito = descargar_con_barra(url, zip_path, descripcion=f"Descarga {split_name}")
            if not exito:
                print(f"[ERROR] Fallo al descargar {split_name}. Verifique enlace.")
                continue

            # 2. Descompresion interactiva con barra
            try:
                descomprimir_con_barra(zip_path, extract_path, password=password, descripcion=f"Extraccion {split_name}")
            finally:
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except Exception:
                        pass

            rutas_extraidas[split_name] = extract_path

    print("[*] Datasets preparados exitosamente.")
    return rutas_extraidas

def ensure_dataset(dataset_dir: str, gdown_id: str = "") -> str:
    """Compatibilidad con modulo general de descarga por gdown."""
    os.makedirs(dataset_dir, exist_ok=True)
    if any(os.scandir(dataset_dir)):
        logger.info(f"[DATASET] Dataset already found in: {dataset_dir}")
        return dataset_dir

    if not gdown_id:
        logger.warning(f"[WARNING] No gdown_id provided and directory is empty: {dataset_dir}")
        return dataset_dir

    try:
        import gdown
    except ImportError:
        logger.error("[ERROR] 'gdown' package is not installed.")
        return dataset_dir

    zip_dest = os.path.join(dataset_dir, "dataset_temp.zip")
    logger.info(f"[DOWNLOAD] Downloading dataset from Google Drive (ID: {gdown_id})...")
    gdown.download(id=gdown_id, output=zip_dest, quiet=False)

    if os.path.exists(zip_dest):
        descomprimir_archivo(zip_dest, dataset_dir)
        if os.path.exists(zip_dest):
            os.remove(zip_dest)
        logger.info("[SUCCESS] Dataset downloaded and extracted successfully.")

    return dataset_dir
