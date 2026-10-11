import os
import subprocess
import zipfile
from typing import Optional
from common.logger import setup_logger

logger = setup_logger("Recomendador-Downloader")

def ejecutar_comando(comando: str):
    subprocess.run(comando, shell=True, check=True)

def descargar_y_extraer_polyvore(url: str, temp_path: str) -> str:
    """
    Descarga Re-PolyVore.zip y lo descomprime en el almacenamiento temporal.
    Retorna la ruta al directorio con el dataset extraido.
    """
    zips_dir = os.path.join(temp_path, "zips")
    data_dir = os.path.join(temp_path, "dataset", "polyvore")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Verificar si ya existen archivos extraidos
    if any(os.scandir(data_dir)):
        print(f"[*] Dataset PolyVore ya detectado en: {data_dir}")
        return data_dir

    if not url:
        print("[!] No se proporciono enlace de descarga. Verifique --link_polyvore.")
        return data_dir

    zip_dest = os.path.join(zips_dir, "re_polyvore.zip")
    print("[*] Descargando Re-PolyVore dataset...")

    descarga_exitosa = False
    try:
        # Descarga con wget respetando comillas por parametros de SharePoint
        ejecutar_comando(f'wget -q -O "{zip_dest}" "{url}"')
        descarga_exitosa = True
    except Exception as e:
        print(f"[!] Wget no disponible o fallo ({e}). Intentando con curl...")
        try:
            ejecutar_comando(f'curl -s -L -o "{zip_dest}" "{url}"')
            descarga_exitosa = True
        except Exception as e2:
            print(f"[!] No se pudo descargar el dataset con curl: {e2}")

    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 0:
        if zipfile.is_zipfile(zip_dest):
            print(f"[*] Descomprimiendo Re-PolyVore en {data_dir}...")
            try:
                ejecutar_comando(f'unzip -q "{zip_dest}" -d "{data_dir}"')
            except Exception:
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    zf.extractall(data_dir)
            print("[*] Re-PolyVore preparado exitosamente.")
        else:
            print("[!] El archivo descargado no es un zip valido o requiere sesion. Verifique el enlace.")
        
        try:
            os.remove(zip_dest)
        except Exception:
            pass

    return data_dir
