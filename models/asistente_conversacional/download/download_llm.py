import os
import subprocess
import argparse

DEFAULT_REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF" # Placeholder descargable o repositorio oficial BellAI
DEFAULT_FILENAME = "bellai_assistant_8b.gguf"
DEFAULT_DRIVE_DEST = "/content/drive/MyDrive/Modelos_BellAI/asistente_conversacional"

def descargar_modelo_gguf(repo_id: str, filename: str, dest_dir: str):
    """
    Descarga el archivo GGUF desde Hugging Face y lo guarda en Google Drive.
    """
    os.makedirs(dest_dir, exist_ok=True)
    target_path = os.path.join(dest_dir, filename)

    if os.path.exists(target_path):
        print(f"[*] Modelo GGUF ya presente en: {target_path}")
        return target_path

    print(f"[*] Descargando {filename} desde Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
        print(f"[*] Descarga completada: {downloaded}")
        return downloaded
    except ImportError:
        print("[!] huggingface_hub no esta instalado. Instalando...")
        subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
        return downloaded
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el modelo GGUF ({e}).")
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargador de LLM GGUF para Asistente Conversacional")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Repositorio Hugging Face")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Nombre del archivo .gguf")
    parser.add_argument("--dest_dir", type=str, default=DEFAULT_DRIVE_DEST, help="Directorio destino en Google Drive")
    args = parser.parse_args()

    descargar_modelo_gguf(args.repo_id, args.filename, args.dest_dir)
