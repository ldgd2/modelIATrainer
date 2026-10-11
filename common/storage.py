import os
import shutil
import torch
from datetime import datetime
from common.logger import setup_logger

logger = setup_logger("Storage")

class StorageManager:
    def __init__(self, base_drive_path: str, temp_path: str):
        self.drive_path = base_drive_path
        self.temp_path = temp_path

        self.checkpoints_dir = os.path.join(self.drive_path, "checkpoints")
        self.releases_dir = os.path.join(self.drive_path, "releases")
        self.logs_dir = os.path.join(self.drive_path, "logs")

    def setup_directories(self):
        """Crea la estructura de carpetas en Google Drive y disco temporal."""
        print(f"[*] Verificando estructura en Drive: {self.drive_path}")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.releases_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)

    def save_checkpoint(self, model, filename: str):
        """Guarda un estado temporal del modelo en Google Drive."""
        filepath = os.path.join(self.checkpoints_dir, filename)
        state = model.state_dict() if hasattr(model, "state_dict") else model
        torch.save(state, filepath)
        print(f"[*] Checkpoint respaldado en Drive: {filepath}")

    def save_release(self, model, version_name: str):
        """Genera el modelo final tanto con el nombre estandar como con timestamp."""
        state = model.state_dict() if hasattr(model, "state_dict") else model

        # 1. Archivo estandar directo (ej. wardrobe_v1.pt)
        clean_name = version_name if version_name.endswith(".pt") else f"{version_name}.pt"
        std_filepath = os.path.join(self.releases_dir, clean_name)
        torch.save(state, std_filepath)

        # 2. Archivo historico con timestamp
        base_name = clean_name[:-3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        time_filepath = os.path.join(self.releases_dir, f"{base_name}_{timestamp}.pt")
        torch.save(state, time_filepath)

        # 3. Exportacion optimizada para telefonos moviles (PyTorch Mobile / TorchScript / Android / iOS)
        if isinstance(model, torch.nn.Module):
            try:
                model_eval = model.cpu().eval()
                if hasattr(model_eval, "e1") or hasattr(model_eval, "initial"):
                    dummy = torch.randn(1, 6, 128, 128)
                elif hasattr(model_eval, "extract_features"):
                    dummy = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
                else:
                    dummy = torch.randn(1, 3, 224, 224)

                with torch.no_grad():
                    if isinstance(dummy, tuple):
                        traced = torch.jit.trace(model_eval, dummy)
                    else:
                        traced = torch.jit.trace(model_eval, dummy)

                    try:
                        from torch.utils.mobile_optimizer import optimize_for_mobile
                        mobile_opt = optimize_for_mobile(traced)
                        mobile_path = os.path.join(self.releases_dir, f"{base_name}_mobile.ptl")
                        if hasattr(mobile_opt, "_save_for_lite_interpreter"):
                            mobile_opt._save_for_lite_interpreter(mobile_path)
                        else:
                            torch.jit.save(mobile_opt, mobile_path)
                        print(f"[*] MODELO MOVIL (Lite Interpreter Android/iOS): {mobile_path}")
                    except Exception:
                        traced_path = os.path.join(self.releases_dir, f"{base_name}_traced.pt")
                        torch.jit.save(traced, traced_path)
                        print(f"[*] MODELO TORCHSCRIPT PARA MOVIL: {traced_path}")
            except Exception:
                pass

        print(f"[*] RELEASE FINAL GENERADO: {std_filepath}")

def save_checkpoint(
    state_dict: dict,
    local_path: str,
    drive_dir: str = None,
    is_best: bool = False
) -> None:
    """Helper de compatibilidad para sincronizacion de checkpoints."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save(state_dict, local_path)
    logger.info(f"Checkpoint saved locally at: {local_path}")

    if drive_dir and os.path.exists(drive_dir):
        os.makedirs(drive_dir, exist_ok=True)
        filename = os.path.basename(local_path)
        dest_path = os.path.join(drive_dir, filename)
        shutil.copy2(local_path, dest_path)
        logger.info(f"Checkpoint synced to Google Drive: {dest_path}")

        if is_best:
            best_dest = os.path.join(drive_dir, "best_model.pt")
            shutil.copy2(local_path, best_dest)
            logger.info(f"Best model synced to Google Drive: {best_dest}")

def find_latest_checkpoint(drive_module_dir: str):
    """
    Busca automaticamente el checkpoint mas reciente en Google Drive o local.
    Retorna (ruta_archivo, ultima_epoca_completada) o (None, 0).
    """
    search_dirs = [
        os.path.join(drive_module_dir, "checkpoints"),
        drive_module_dir
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pt") and ("epoch" in f.lower() or "latest" in f.lower() or "model" in f.lower()):
                    candidates.append(os.path.join(d, f))

    if not candidates:
        return None, 0

    import re
    best_file = None
    max_epoch = 0

    for path in candidates:
        match = re.search(r"epoch_?(\d+)", os.path.basename(path), re.IGNORECASE)
        if match:
            ep = int(match.group(1))
            if ep > max_epoch:
                max_epoch = ep
                best_file = path

    if not best_file:
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        best_file = candidates[0]

    if best_file:
        try:
            state = torch.load(best_file, map_location="cpu")
            if isinstance(state, dict) and "epoch" in state:
                max_epoch = max(max_epoch, int(state["epoch"]))
        except Exception:
            pass

    return best_file, max_epoch

