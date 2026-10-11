import os
import subprocess
from typing import Optional
from common.logger import setup_logger

logger = setup_logger("Kaggle-Utils")

DEFAULT_KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")

# Catalogo oficial de datasets de BellAI
CATALOGO_DATASETS = {
    # 1. Armario Digital (Dataset de prendas y categorias publico libre)
    "deepfashion2": "agrigorev/clothing-dataset-full",

    # 2. Motor de Recomendacion
    "polyvore": "Re-PolyVore",

    # 3. Colorimetria Facial
    "fairface": "aibloy/fairface",

    # 4. Proporciones y Silueta Corporal
    "coco_keypoints": "asad11914/coco-2017-keypoints",

    # 5. Probador Virtual (Mapeo de prendas)
    "viton_hd": "marquis03/high-resolution-viton-zalando-dataset",

    # 6. Analisis Capilar
    "celebamaskhq": "ipythonx/celebamaskhq",

    # 7. Simulacion de Maquillaje
    "mt_dataset": "cyd0806/makeup-transfer-dataset"
}

def configurar_kaggle(token: Optional[str] = None) -> bool:
    """
    Configura el token de acceso para la API de Kaggle y kagglehub.
    Exporta KAGGLE_API_TOKEN y genera ~/.kaggle/access_token con permisos 600.
    """
    token_activo = token or os.environ.get("KAGGLE_API_TOKEN") or DEFAULT_KAGGLE_TOKEN

    if not token_activo:
        print("[!] No se encontro un token de Kaggle configurado.")
        return False

    os.environ["KAGGLE_API_TOKEN"] = token_activo

    kaggle_dir = os.path.expanduser("~/.kaggle")
    token_file = os.path.join(kaggle_dir, "access_token")

    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(token_activo.strip() + "\n")

        if hasattr(os, "chmod"):
            try:
                os.chmod(token_file, 0o600)
            except Exception:
                pass

        print("[*] Autenticacion de Kaggle configurada exitosamente.")
        return True
    except Exception as e:
        print(f"[!] Advertencia al escribir ~/.kaggle/access_token ({e}).")
        return True

def descargar_con_kagglehub(slug_o_alias: str) -> str:
    """
    Descarga la version mas reciente de un dataset usando kagglehub.
    Acepta el slug directo (ej. 'aibloy/fairface') o un alias del catalogo.
    """
    configurar_kaggle()
    slug = CATALOGO_DATASETS.get(slug_o_alias, slug_o_alias)

    try:
        import kagglehub
        print(f"[*] Descargando via kagglehub: {slug}...")
        path = kagglehub.dataset_download(slug)
        print(f"[*] Dataset disponible en: {path}")
        return path
    except ImportError:
        print("[!] kagglehub no esta instalado. Instalando...")
        subprocess.run("pip install -q kagglehub", shell=True, check=True)
        import kagglehub
        return kagglehub.dataset_download(slug)
    except Exception as e:
        print(f"[ERROR] Error al descargar con kagglehub ({e}).")
        if "deepfashion" in slug.lower() or "nadyapavlova" in slug.lower():
            fallback_slug = "agrigorev/clothing-dataset-full"
            print(f"[*] Intentando con dataset publico abierto ({fallback_slug})...")
            try:
                import kagglehub
                return kagglehub.dataset_download(fallback_slug)
            except Exception as e2:
                print(f"[ERROR] Error en fallback ({e2}).")
        return ""

def descargar_dataset_kaggle(slug: str, dest_dir: str, token: Optional[str] = None) -> str:
    """Descarga y descomprime un dataset usando la CLI tradicional de Kaggle."""
    configurar_kaggle(token)
    os.makedirs(dest_dir, exist_ok=True)

    if any(os.scandir(dest_dir)):
        print(f"[*] Dataset de Kaggle ya presente en: {dest_dir}")
        return dest_dir

    print(f"[*] Descargando dataset desde Kaggle CLI: {slug}...")
    cmd = f'kaggle datasets download -d "{slug}" -p "{dest_dir}" --unzip'

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[*] Dataset {slug} descargado y descomprimido en {dest_dir}.")
    except Exception as e:
        print(f"[ERROR] Fallo al descargar dataset desde Kaggle ({e}).")

    return dest_dir
