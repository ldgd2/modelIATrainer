# Template para Nuevo Modelo de Entrenamiento ($namemodel)

Para crear un nuevo modelo modular (ejemplo: `maquillaje`, `rostros`, etc.):

1. Duplica esta carpeta `_template` y renómbrala con el nombre de tu modelo:
   ```bash
   cp -r models/_template models/mi_nuevo_modelo
   ```
2. Edita `config/config.yaml` con tus hiperparámetros por defecto.
3. Ajusta `architecture/network.py` y `dataset/transforms.py` a tus requerimientos.
4. Ejecuta desde Colab o localmente:
   ```bash
   python models/mi_nuevo_modelo/main.py --dataset /content/dataset/ --batch_size 16 --epochs 20
   ```
