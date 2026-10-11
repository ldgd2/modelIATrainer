"""
Orquestador Unico de Entrenamiento para BellAI.
Delega la ejecucion al modulo correspondiente dentro de models/<modulo>/main.py
"""
import sys
import os
import argparse
import subprocess

MODULOS_DISPONIBLES = [
    "armario_digital",
    "diagnostico_morfologico",
    "motor_recomendacion",
    "simulacion_virtual",
    "asistente_conversacional"
]

def main():
    parser = argparse.ArgumentParser(
        description="BellAI - Orquestador Unico de Entrenamiento",
        add_help=False
    )
    parser.add_argument(
        "--module",
        type=str,
        default="armario_digital",
        choices=MODULOS_DISPONIBLES,
        help="Modulo a ejecutar"
    )

    # Extraer el argumento --module y pasar el resto al script del modulo
    args, rest_args = parser.parse_known_args()

    module_script = os.path.join("models", args.module, "main.py")
    if not os.path.exists(module_script):
        print(f"[ERROR] Script no encontrado: {module_script}")
        sys.exit(1)

    cmd = [sys.executable, module_script] + rest_args
    print(f"[*] Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
