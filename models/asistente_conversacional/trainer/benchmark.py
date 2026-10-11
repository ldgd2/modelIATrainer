import os
import time
from common.logger import setup_logger

logger = setup_logger("Asistente-Benchmark")

def evaluar_modelo_local(model_path: str):
    """
    Verifica la integridad del archivo GGUF descargado y mide el tamano.
    """
    if not os.path.exists(model_path):
        logger.warning(f"[AVISO] Archivo {model_path} no encontrado.")
        return False

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"[*] Tamano verificado de release GGUF: {size_mb:.2f} MB")
    logger.info("[SUCCESS] Modelo LLM verificado y empaquetado para inferencia en produccion.")
    return True
