import os
import subprocess
import zipfile
from typing import Optional
from common.logger import setup_logger
from common.kaggle_utils import descargar_dataset_kaggle, descargar_con_kagglehub

logger = setup_logger("Colorimetria-Downloader")

FAIRFACE_KAGGLE_SLUG = "aibloy/fairface"
CELEBAMASK_KAGGLE_SLUG = "ipythonx/celebamaskhq"

def ejecutar_comando(comando: str):
    subprocess.run(comando, shell=True, check=True)

def descargar_fairface_directo(url: str, dest_dir: str) -> str:
    """Descarga el bundle de FairFace usando el enlace directo de Google Storage."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "fairface_archive.zip")

    print("[*] Descargando FairFace desde enlace directo de Google Storage...")
    try:
        ejecutar_comando(f'wget -q -O "{zip_path}" "{url}"')
    except Exception:
        ejecutar_comando(f'curl -s -L -o "{zip_path}" "{url}"')

    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        print(f"[*] Descomprimiendo FairFace en {dest_dir}...")
        try:
            ejecutar_comando(f'unzip -q "{zip_path}" -d "{dest_dir}"')
        except Exception:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        try:
            os.remove(zip_path)
        except Exception:
            pass
        print("[*] FairFace preparado exitosamente.")

    return dest_dir

def ensure_dataset(
    dataset_dir: str,
    gdown_id: str = "",
    kaggle_slug: str = "",
    direct_url: str = "",
    usar_kagglehub: bool = True
) -> str:
    """
    Verifica o descarga el dataset de colorimetria facial.
    Prioridades:
    1. Si ya hay imagenes en dataset_dir, retorna la ruta.
    2. Si se activa usar_kagglehub, descarga 'aibloy/fairface' o el slug provisto.
    3. Si se proporciona direct_url, descarga via wget/curl.
    4. Fallback a gdown_id.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    if any(os.scandir(dataset_dir)):
        logger.info(f"[DATASET] Dataset encontrado en: {dataset_dir}")
        return dataset_dir

    # 1. Descarga moderna con kagglehub
    slug = kaggle_slug or FAIRFACE_KAGGLE_SLUG
    if usar_kagglehub:
        path = descargar_con_kagglehub(slug)
        if path and os.path.exists(path):
            return path

    # 2. Descarga por URL directa de Google Storage
    if direct_url:
        return descargar_fairface_directo(direct_url, dataset_dir)

    # 3. Fallback a Kaggle CLI tradicional
    if kaggle_slug:
        return descargar_dataset_kaggle(kaggle_slug, dataset_dir)

    # 4. Fallback a Google Drive (gdown)
    if gdown_id:
        try:
            import gdown
            zip_dest = os.path.join(dataset_dir, "dataset_temp.zip")
            logger.info(f"[DOWNLOAD] Descargando desde Google Drive (ID: {gdown_id})...")
            gdown.download(id=gdown_id, output=zip_dest, quiet=False)
            if os.path.exists(zip_dest):
                with zipfile.ZipFile(zip_dest, "r") as zip_ref:
                    zip_ref.extractall(dataset_dir)
                os.remove(zip_dest)
                return dataset_dir
        except Exception as e:
            logger.error(f"[ERROR] Fallo descarga gdown: {e}")

    logger.warning(f"[WARNING] No se configuro fuente de datos valida para {dataset_dir}.")
    return dataset_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Descargador de Dataset para Colorimetria")
    parser.add_argument("--dest", type=str, default="data/colorimetria")
    parser.add_argument("--kaggle_slug", type=str, default=FAIRFACE_KAGGLE_SLUG)
    parser.add_argument("--direct_url", type=str, default="")
    args = parser.parse_args()

    ensure_dataset(args.dest, kaggle_slug=args.kaggle_slug, direct_url=args.direct_url)
