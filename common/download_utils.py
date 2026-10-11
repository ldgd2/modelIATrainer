import os
import sys
import subprocess
import zipfile
from typing import Optional

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, *args, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])
        def set_postfix(self, *args, **kwargs):
            pass

def descargar_con_barra(url: str, destino: str, descripcion: str = "Descarga", forzar: bool = False) -> bool:
    """
    Descarga un archivo mostrando una barra de progreso interactiva.
    Si el archivo ya existe y es valido, omite la descarga automaticamente.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destino)), exist_ok=True)
    nombre_archivo = os.path.basename(destino)

    # Control de existencia previa: evitar descargas redundantes
    if not forzar and os.path.exists(destino) and os.path.getsize(destino) > 1024:
        mb = os.path.getsize(destino) / (1024 * 1024)
        print(f"[*] Archivo ya existente en disco: {nombre_archivo} ({mb:.2f} MB). Omitiendo descarga.")
        return True

    try:
        import requests
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))

        chunk_size = 1024 * 1024  # 1 MB por bloque
        print(f"[*] Iniciando descarga interactiva: {nombre_archivo}")

        with open(destino, "wb") as f, tqdm(
            desc=f"{descripcion} [{nombre_archivo}]",
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=100,
            leave=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        if os.path.exists(destino) and os.path.getsize(destino) > 0:
            return True

    except Exception as e:
        print(f"[INFO] Motor requests no pudo completar descarga directa ({e}). Intentando con wget/curl con barra...")

    # Fallback 1: wget con barra de progreso interactiva (estandar en Colab / Linux)
    try:
        cmd = f'wget --show-progress -c -O "{destino}" "{url}"'
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(destino) and os.path.getsize(destino) > 0:
            return True
    except Exception:
        pass

    # Fallback 2: curl con barra interactiva
    try:
        cmd = f'curl -# -L -o "{destino}" "{url}"'
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(destino) and os.path.getsize(destino) > 0:
            return True
    except Exception as e3:
        print(f"[ERROR] No se pudo descargar el archivo: {e3}")
        return False

    return False

def descomprimir_con_barra(zip_path: str, extract_dir: str, password: Optional[str] = None, descripcion: str = "Extrayendo") -> bool:
    """
    Descomprime un archivo zip mostrando una barra animada con el total de archivos,
    porcentaje y velocidad de extraccion.
    """
    os.makedirs(extract_dir, exist_ok=True)
    pwd_bytes = password.encode("utf-8") if password else None

    # Si el sistema tiene unzip (Linux / Colab) y es un archivo muy grande con password
    if sys.platform != "win32":
        try:
            print(f"[*] Descomprimiendo {os.path.basename(zip_path)}...")
            if password:
                cmd = f'unzip -P "{password}" "{zip_path}" -d "{extract_dir}" | pv -l -s 50000 > /dev/null'
            else:
                cmd = f'unzip "{zip_path}" -d "{extract_dir}" | pv -l -s 50000 > /dev/null'
            res = subprocess.run(cmd, shell=True)
            if res.returncode == 0:
                return True
        except Exception:
            pass

    # Extraccion nativa con Python zipfile y barra tqdm de progreso
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            miembros = zf.infolist()
            total_archivos = len(miembros)

            with tqdm(total=total_archivos, desc=f"{descripcion} [{os.path.basename(zip_path)}]", unit="archivos", ncols=100, leave=True) as pbar:
                for miembro in miembros:
                    try:
                        zf.extract(miembro, path=extract_dir, pwd=pwd_bytes)
                    except Exception:
                        pass
                    pbar.update(1)

            print(f"[*] Extraccion completada exitosamente en: {extract_dir}")
            return True
    except Exception as e:
        print(f"[ERROR] Error al descomprimir archivo {zip_path}: {e}")
        return False
