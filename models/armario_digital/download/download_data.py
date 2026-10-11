import os
import subprocess
import zipfile
from typing import Dict, Optional
from common.logger import setup_logger

logger = setup_logger("Armario-Downloader")

def ejecutar_comando(comando: str):
    """Ejecuta un comando en shell verificando errores."""
    subprocess.run(comando, shell=True, check=True)

def descomprimir_archivo(zip_path: str, extract_path: str, password: Optional[str] = "2019Deepfashion2**"):
    """
    Descomprime un archivo zip usando unzip con password si esta disponible,
    o con fallback a zipfile nativo de Python.
    """
    os.makedirs(extract_path, exist_ok=True)
    pwd_bytes = password.encode("utf-8") if password else None

    # Intento 1: comando unzip del sistema (ideal para Linux / Google Colab)
    try:
        if password:
            cmd = f'unzip -q -P "{password}" "{zip_path}" -d "{extract_path}"'
        else:
            cmd = f'unzip -q "{zip_path}" -d "{extract_path}"'
        ejecutar_comando(cmd)
        return
    except Exception as e:
        print(f"[INFO] Comando unzip no disponible o fallo ({e}). Usando motor nativo zipfile...")

    # Intento 2: zipfile nativo de Python
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_path, pwd=pwd_bytes)

def preparar_dataset(links: Dict[str, str], temp_path: str, password: Optional[str] = "2019Deepfashion2**") -> Dict[str, str]:
    """
    Descarga y descomprime los datasets en el almacenamiento efimero.
    Maneja el password para DeepFashion2 (2019Deepfashion2**).
    Retorna un diccionario con las rutas locales de los datos extraidos.
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

            print(f"[*] Descargando {split_name}...")
            # Uso de comillas dobles para evitar que bash rompa la URL por el caracter '&'
            try:
                ejecutar_comando(f'wget -q -O "{zip_path}" "{url}"')
            except Exception as e:
                print(f"[ERROR] Fallo al descargar con wget ({e}). Verifique enlace o conexion.")
                continue

            print(f"[*] Descomprimiendo {split_name} (con password si aplica)...")
            try:
                descomprimir_archivo(zip_path, extract_path, password=password)
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
