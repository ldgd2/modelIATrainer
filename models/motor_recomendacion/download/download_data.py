import os
import subprocess
import zipfile
from typing import Optional
from common.logger import setup_logger

from common.download_utils import descargar_con_barra, descomprimir_con_barra

logger = setup_logger("Recomendador-Downloader")

def descargar_y_extraer_polyvore(url: str, temp_path: str) -> str:
    """
    Descarga Re-PolyVore.zip y lo descomprime mostrando barra animada con
    porcentaje, MB descargados/totales y tiempo estimado.
    """
    zips_dir = os.path.join(temp_path, "zips")
    data_dir = os.path.join(temp_path, "dataset", "polyvore")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if any(os.scandir(data_dir)):
        print(f"[*] Dataset PolyVore ya detectado en: {data_dir}")
        return data_dir

    if not url:
        print("[!] No se proporciono enlace de descarga. Verifique --link_polyvore.")
        return data_dir

    zip_dest = os.path.join(zips_dir, "re_polyvore.zip")

    # 1. Descarga con barra interactiva
    exito = descargar_con_barra(url, zip_dest, descripcion="Descarga PolyVore")
    if not exito:
        print("[ERROR] No se pudo descargar el dataset PolyVore.")
        return data_dir

    # 2. Descompresion con barra interactiva
    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 0:
        if zipfile.is_zipfile(zip_dest):
            descomprimir_con_barra(zip_dest, data_dir, descripcion="Extraccion PolyVore")
            print("[*] Re-PolyVore preparado exitosamente.")
        else:
            print("[!] El archivo descargado no es un zip valido o requiere sesion. Verifique el enlace.")

        try:
            os.remove(zip_dest)
        except Exception:
            pass

    return data_dir
