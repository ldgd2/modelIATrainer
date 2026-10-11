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

    module_name = f"models.{args.module}.main"
    sys.argv = [os.path.join("models", args.module, "main.py")] + rest_args

    print(f"[*] Lanzando modulo: {args.module}")
    print(f"[*] Parametros: {' '.join(rest_args) if rest_args else 'Valores por defecto'}\n")

    import importlib
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"[ERROR] El modulo {module_name} no tiene funcion main().")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fallo en la ejecucion del modulo {args.module}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
