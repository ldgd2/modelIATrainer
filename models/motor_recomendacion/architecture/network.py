import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class StyleRecommenderNet(nn.Module):
    """
    Red neuronal siamesa para calificar compatibilidad estetica entre pares de prendas.
    Proyecta ambas prendas a un espacio latente comun de estilo y calcula similitud coseno.
    """
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

    def extract_features(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

    def forward(self, item_a, item_b=None):
        emb_a = self.extract_features(item_a)
        if item_b is None:
            return emb_a
        emb_b = self.extract_features(item_b)
        sim = F.cosine_similarity(emb_a, emb_b)
        return (sim + 1.0) / 2.0  # Normalizado en rango [0, 1]
