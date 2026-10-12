import os
import subprocess
import zipfile
from typing import Optional
from common.logger import setup_logger

from common.download_utils import descargar_con_barra, descomprimir_con_barra

logger = setup_logger("Recomendador-Downloader")

def descargar_polyvore_kaggle(slug: str = "nastarang/re-polyvore") -> str:
    """Descarga Re-PolyVore directamente desde Kaggle via kagglehub."""
    from common.kaggle_utils import descargar_con_kagglehub
    logger.info(f"[*] Descargando dataset PolyVore via kagglehub ({slug})...")
    path = descargar_con_kagglehub(slug)
    return path

def descargar_y_extraer_polyvore(url: str, temp_path: str, kaggle_slug: str = "nastarang/re-polyvore") -> str:
    """
    Descarga Re-PolyVore.zip y lo descomprime mostrando barra animada con
    porcentaje, MB descargados/totales y tiempo estimado.
    Si el enlace directo falla (ej. 403 Forbidden), intenta con Kagglehub.
    """
    zips_dir = os.path.join(temp_path, "zips")
    data_dir = os.path.join(temp_path, "dataset", "polyvore")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if any(os.scandir(data_dir)):
        print(f"[*] Dataset PolyVore ya detectado en: {data_dir}")
        return data_dir

    # Intentar primero con la URL proporcionada
    zip_dest = os.path.join(zips_dir, "re_polyvore.zip")
    exito = False
    if url and not url.startswith("none"):
        exito = descargar_con_barra(url, zip_dest, descripcion="Descarga PolyVore")

    # Si la URL falla (ej. 403 Forbidden) o no se proporciono, recurrir a Kagglehub
    if not exito:
        print(f"[*] Enlace directo no disponible o bloqueado. Recurriendo a Kagglehub ({kaggle_slug})...")
        k_path = descargar_polyvore_kaggle(kaggle_slug)
        if k_path and os.path.exists(k_path):
            return k_path

    # Descompresion con barra interactiva si se descargo el zip
    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 0:
        if zipfile.is_zipfile(zip_dest):
            descomprimir_con_barra(zip_dest, data_dir, descripcion="Extraccion PolyVore")
            print("[*] Re-PolyVore preparado exitosamente.")
        else:
            print("[!] El archivo descargado no es un zip valido.")

        try:
            os.remove(zip_dest)
        except Exception:
            pass

    return data_dir
