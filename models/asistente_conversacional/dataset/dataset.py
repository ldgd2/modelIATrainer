import os
import json
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from common.logger import setup_logger

logger = setup_logger("Asistente-Dataset")

# Dataset especializado en Asesoria de Imagen, Estilismo y Colorimetria BellAI
BELLAI_FASHION_INSTRUCTIONS = [
    {
        "instruction": "Tengo piel con subtono frio, ojos claros y cabello castano oscuro. Que paleta de colorimetria me favorece?",
        "response": "Tu armonia corresponde a la estacion Invierno. Te favorecen los tonos frios y de alto contraste como azul marino, negro profundo, blanco puro, fucsia, esmeralda y plateado. Evita tonos tierra o dorados que apaguen tu luminosidad."
    },
    {
        "instruction": "Como puedo combinar un blazer negro oversize para un look casual?",
        "response": "Puedes combinarlo con una camiseta basica blanca, jeans de corte recto o wide-leg claros y zapatillas blancas. Agrega accesorios dorados sutiles y un bolso cruzado para equilibrar la formalidad del blazer."
    },
    {
        "instruction": "Tengo cuerpo tipo triangulo invertido (hombros mas anchos que caderas). Que prendas equilibran mi silueta?",
        "response": "El objetivo es dar volumen a la parte inferior y suavizar los hombros. Usa pantalones palazzo, faldas acampanadas (linea A) o cargo, y en la parte superior prendas lisas con escote en V o asimetrico, evitando hombreras o mangas abullonadas."
    },
    {
        "instruction": "Que colores van mejor con una falda satinada beige champan?",
        "response": "Combina de forma sofisticada con tonos neutros: un jersey de punto blanco roto, blusas en verde salvia, marron chocolate o negro para contraste. En calzado, botines nude o sandalias minimalistas."
    },
    {
        "instruction": "Cual es la diferencia entre un cuerpo reloj de arena y rectangulo?",
        "response": "El cuerpo reloj de arena presenta hombros y caderas alineados con una cintura notablemente definida. El cuerpo rectangulo tiene medidas similares entre hombros, cintura y cadera con poca definicion en la cintura, donde se busca crear curvas mediante cinturones y cortes peplum."
    },
    {
        "instruction": "Que zapatos llevar con un vestido midi ajustado?",
        "response": "Para estilizar, stilettos o sandalias de tacon fino en tonos neutros alargan las piernas. Para un look urbano y comodo, zapatillas retro blancas o botines de cana baja con chaqueta biker de cuero."
    },
    {
        "instruction": "Cuales son los 5 basicos imprescindibles para un armario capsula inteligente?",
        "response": "1. Camisa blanca de corte clasico. 2. Jeans rectos de lavado medio. 3. Blazer estructurado en tono neutro (negro o camel). 4. Vestido sencillo versatil (LBD). 5. Abrigo de lana largo o gabardina impermeable."
    },
    {
        "instruction": "Que tonos favorecen a la estacion Primavera Brillante?",
        "response": "Colores calidos, vivos y luminosos: coral vibrante, turquesa, amarillo calido, verde manzana y tonos dorados. Evita tonos apagados, grises oscuros o colores excesivamente frios."
    },
    {
        "instruction": "Como disimular volumen en la zona abdominal sin perder estilo?",
        "response": "Opta por pantalones o faldas de tiro alto con tejidos estructurados, vestidos de corte imperio o cruzados (wrap dress), y blusas con drapeados suaves. Las terceras capas abiertas como blazers largos crean lineas verticales estilizadoras."
    },
    {
        "instruction": "Que outfit recomiendas para una entrevista de trabajo en una empresa creativa?",
        "response": "Un traje sastre de dos piezas en un tono contemporaneo como verde oliva o terracota, combinado con camiseta blanca de calidad, mocasines elegantes y accesorios minimalistas. Transmite profesionalismo con personalidad."
    }
]

class StylistInstructionDataset(Dataset):
    """
    Dataset para fine-tuning del asistente de estilismo BellAI.
    Soporta datasets personalizados JSON/JSONL o el corpus de moda integrado.
    """
    def __init__(self, data_path: str = None, tokenizer = None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict[str, str]] = []

        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    if data_path.endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                self.data.append(json.loads(line))
                    else:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            self.data = loaded
                logger.info(f"[DATASET] Cargadas {len(self.data)} instrucciones desde {data_path}.")
            except Exception as e:
                logger.warning(f"Error al leer dataset personalizado ({e}). Usando corpus integrado.")
                self.data = BELLAI_FASHION_INSTRUCTIONS
        else:
            self.data = BELLAI_FASHION_INSTRUCTIONS
            logger.info(f"[DATASET] Usando corpus base de estilismo BellAI ({len(self.data)} muestras).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", item.get("prompt", ""))
        response = item.get("response", item.get("completion", ""))

        prompt_text = (
            f"<|system|>\nEres BellAI Stylist, asistente de IA experta en moda, estilismo, colorimetria y morfologia.\n"
            f"<|user|>\n{instruction}\n"
            f"<|assistant|>\n{response}"
        )

        if self.tokenizer:
            enc = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Fallback en caso de no tener tokenizer de HuggingFace cargado
            # Genera tensores sinteticos representativos
            tokens = torch.randint(100, 30000, (self.max_length,))
            return {
                "input_ids": tokens,
                "attention_mask": torch.ones(self.max_length),
                "labels": tokens
            }

def build_llm_dataloaders(
    data_path: str = None,
    tokenizer = None,
    batch_size: int = 4,
    max_length: int = 256
) -> DataLoader:
    dataset = StylistInstructionDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
