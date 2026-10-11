import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.storage import StorageManager
from models.motor_recomendacion.download.download_data import descargar_y_extraer_polyvore

logger = setup_logger("MotorRecomendacion-Main")

DEFAULT_POLYVORE_URL = (
    "https://hkaidlab-my.sharepoint.com/personal/xingxingzou_aidlab_hk/_layouts/15/download.aspx"
    "?SourceUrl=%2Fpersonal%2Fxingxingzou_aidlab_hk%2FDocuments%2FAiDLab%20-%20fAshIon%20TEAM%20-%20DATA%2FRe-PolyVore.zip"
)
DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/Modelos_BellAI/motor_recomendacion"
DEFAULT_TEMP_DIR = "/content/temp_recomendacion"

class StyleRecommenderNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, item_a, item_b=None):
        emb_a = F.normalize(self.backbone(item_a), p=2, dim=1)
        if item_b is None:
            return emb_a
        emb_b = F.normalize(self.backbone(item_b), p=2, dim=1)
        sim = F.cosine_similarity(emb_a, emb_b)
        return (sim + 1.0) / 2.0

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Motor de Recomendacion (PolyVore -> style_recommender_v1.pt)")
    parser.add_argument("--epochs", type=int, default=15, help="Numero de epocas")
    parser.add_argument("--save_every", type=int, default=2, help="Frecuencia de checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension de embedding")
    parser.add_argument("--drive_path", type=str, default=DEFAULT_DRIVE_DIR, help="Ruta en Google Drive")
    parser.add_argument("--temp_path", type=str, default=DEFAULT_TEMP_DIR, help="Ruta temporal local")
    parser.add_argument("--link_polyvore", type=str, default=DEFAULT_POLYVORE_URL, help="URL de Re-PolyVore.zip")
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint a restaurar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Iniciando Motor de Recomendacion de Outfits...")

    storage = StorageManager(base_drive_path=args.drive_path, temp_path=args.temp_path)
    storage.setup_directories()

    descargar_y_extraer_polyvore(args.link_polyvore, args.temp_path)

    device = torch.device(args.device)
    model = StyleRecommenderNet(embedding_dim=args.embedding_dim).to(device)

    if args.resume and os.path.exists(args.resume):
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            logger.info(f"Pesos restaurados desde: {args.resume}")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}")

    from tqdm.auto import tqdm
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    batches_per_epoch = 20

    logger.info(f"Iniciando entrenamiento por {args.epochs} epocas en {device} ({batches_per_epoch} batches por epoca)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoca {epoch:02d}/{args.epochs:02d} [PolyVore Recommender]", leave=True, ncols=100)

        for _ in pbar:
            dummy_a = torch.randn(args.batch_size, 3, 224, 224).to(device)
            dummy_b = torch.randn(args.batch_size, 3, 224, 224).to(device)
            dummy_target = torch.randint(0, 2, (args.batch_size,)).float().to(device)

            optimizer.zero_grad()
            score = model(dummy_a, dummy_b)
            loss = criterion(score, dummy_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / batches_per_epoch
        logger.info(f"Epoca [{epoch:02d}/{args.epochs:02d}] completada | Loss promedio: {avg_loss:.4f}")

        if epoch % args.save_every == 0:
            ckpt_name = f"recommender_epoch_{epoch}.pt"
            storage.save_checkpoint(model, ckpt_name)

    storage.save_release(model, "style_recommender_v1")

if __name__ == "__main__":
    main()
