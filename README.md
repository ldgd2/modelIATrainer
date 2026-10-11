# ModelTrainingAI - BellAI Training Engine

> Repositorio modular de entrenamiento de Inteligencia Artificial para BellAI, disenado para ejecutarse en Google Colab (o cualquier servidor GPU) mediante terminal y Git, separando completamente el codigo fuente de los cuadernos de computo.

---

## Estructura Oficial en Google Drive (/Modelos_BellAI/)

Todos los checkpoints y releases finales se sincronizan automaticamente en las subcarpetas de tu Google Drive:

```text
/Modelos_BellAI/ (Carpeta raiz en Google Drive)
├── armario_digital/
│   └── wardrobe_v1.pt (Entrenado con DeepFashion2)
├── diagnostico_morfologico/
│   ├── colorimetry_v1.pt (Entrenado con FairFace)
│   ├── morphology_v1.pt (Entrenado con COCO Keypoints)
│   └── hair_analysis_v1.pt (Entrenado con CelebAMask-HQ / Figaro-1k)
├── motor_recomendacion/
│   └── style_recommender_v1.pt (Entrenado con Polyvore)
├── simulacion_virtual/
│   ├── virtual_tryon_v1.pt (Entrenado con VITON-HD)
│   └── makeup_sim_v1.pt (Entrenado con MT-Dataset)
└── asistente_conversacional/
    └── bellai_assistant_8b.gguf (Modelo LLM Hugging Face)
```

---

## Los 5 Modulos del Repositorio

El repositorio cuenta unicamente con **5 modulos estandarizados** dentro de `models/`, cada uno completamente autocontenido con sus submódulos de configuracion, descarga, dataset, arquitectura, entrenador y su script `main.py`:

```text
ModelTrainingAI/
│
├── train.py                              # Orquestador unico de ejecucion en la raiz
│
├── common/                               # Utilidades transversales
│   ├── storage.py                        # StorageManager (gestion de Drive /Modelos_BellAI/)
│   ├── kaggle_utils.py                   # Autenticacion y descargas con kagglehub y Kaggle API
│   ├── config.py                         # Cargador de configuracion YAML y merge con CLI
│   └── logger.py                         # Logger limpio sin emojis
│
├── models/                               # Los 5 modulos oficiales de BellAI
│   │
│   ├── armario_digital/                  # MODULO 1: Deteccion de Prendas
│   │   ├── config/config.yaml
│   │   ├── download/download_data.py     # Descarga DeepFashion2 con password
│   │   ├── dataset/dataset.py
│   │   ├── architecture/network.py
│   │   ├── trainer/trainer.py
│   │   └── main.py                       # Genera: wardrobe_v1.pt
│   │
│   ├── diagnostico_morfologico/          # MODULO 2: Diagnostico Facial y Corporal
│   │   ├── config/config.yaml
│   │   ├── download/download_data.py     # FairFace, COCO Keypoints, CelebAMask-HQ
│   │   ├── dataset/dataset.py
│   │   ├── architecture/network.py
│   │   ├── trainer/trainer.py
│   │   └── main.py                       # Genera: colorimetry_v1.pt, morphology_v1.pt, hair_analysis_v1.pt
│   │
│   ├── motor_recomendacion/              # MODULO 3: Compatibilidad de Outfits
│   │   ├── config/config.yaml
│   │   ├── download/download_data.py     # Descarga Re-PolyVore.zip
│   │   ├── dataset/dataset.py
│   │   ├── architecture/network.py
│   │   ├── trainer/trainer.py
│   │   └── main.py                       # Genera: style_recommender_v1.pt
│   │
│   ├── simulacion_virtual/               # MODULO 4: Probador y Maquillaje
│   │   ├── config/config.yaml
│   │   ├── download/download_data.py     # VITON-HD y MT-Dataset
│   │   ├── dataset/dataset.py
│   │   ├── architecture/network.py
│   │   ├── trainer/trainer.py
│   │   └── main.py                       # Genera: virtual_tryon_v1.pt, makeup_sim_v1.pt
│   │
│   └── asistente_conversacional/         # MODULO 5: LLM Local GGUF
│       ├── config/config.yaml
│       ├── download/download_llm.py      # Descarga GGUF desde Hugging Face
│       └── main.py                       # Entrega: bellai_assistant_8b.gguf
│
├── notebooks/
│   └── colab_launcher.ipynb              # Lanzador interactivo con menu CLI para Colab
│
├── requirements.txt                      # Dependencias del pipeline (torch, kagglehub, etc.)
├── .env.example                          # Plantilla de variables de entorno (privacidad)
├── .gitignore
└── README.md
```

---

## Configuracion de Credenciales Privadas

Por motivos de seguridad, nunca guardes tokens privados en el repositorio publico. Configura tu token de Kaggle de forma generica:

1. Crea o copia tu archivo local `.env`:
   ```bash
   cp .env.example .env
   ```
2. Asigna tu token privado en `.env` o en la consola de Colab:
   ```bash
   export KAGGLE_API_TOKEN="TU_KAGGLE_API_TOKEN_AQUI"
   ```

---

## Ejecucion mediante el Orquestador Unico (train.py)

En lugar de multiples archivos dispersos, utilizas el orquestador unico `train.py` pasandole el modulo deseado:

### 1. Entrenar Armario Digital
```bash
python train.py --module armario_digital --epochs 10 --batch_size 64
```

### 2. Entrenar Diagnostico Morfologico (Subtareas disponibles: colorimetria, morfologia, capilar)
```bash
# Colorimetria (FairFace -> colorimetry_v1.pt)
python train.py --module diagnostico_morfologico --task colorimetria --epochs 20 --batch_size 32 --download_data

# Silueta Corporal (COCO Keypoints -> morphology_v1.pt)
python train.py --module diagnostico_morfologico --task morfologia --epochs 10 --download_data

# Analisis Capilar (CelebAMask-HQ -> hair_analysis_v1.pt)
python train.py --module diagnostico_morfologico --task capilar --epochs 10 --download_data
```

### 3. Entrenar Motor de Recomendacion
```bash
python train.py --module motor_recomendacion --epochs 15 --batch_size 32
```

### 4. Entrenar Simulacion Virtual (Subtareas: probador, maquillaje)
```bash
# Probador Virtual (VITON-HD -> virtual_tryon_v1.pt)
python train.py --module simulacion_virtual --task probador --epochs 10 --download_data

# Simulacion de Maquillaje (MT-Dataset -> makeup_sim_v1.pt)
python train.py --module simulacion_virtual --task maquillaje --epochs 10 --download_data
```

### 5. Descargar Asistente Conversacional (GGUF)
```bash
python train.py --module asistente_conversacional
```

Tambien puedes ejecutar directamente el archivo `main.py` dentro de cada carpeta:
```bash
python models/armario_digital/main.py --epochs 10
```

---

## Ejecucion Interactiva en Google Colab

Abre [`notebooks/colab_launcher.ipynb`](notebooks/colab_launcher.ipynb) y ejecuta su celda.
Te mostrara el menu interactivo, te solicitara tu token privado unicamente en memoria de ejecucion y te permitira restaurar cualquier checkpoint existente en tu Drive con `--resume`.
