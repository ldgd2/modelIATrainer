import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)

class StyleRecommenderNet(nn.Module):
    """
    Red neuronal siamesa para calificar compatibilidad estetica entre pares de prendas.
    Proyecta ambas prendas a un espacio latente comun de estilo y calcula similitud coseno.
    Soporta MobileNet para inferencia ultrarrapida en telefonos moviles.
    """
    def __init__(self, embedding_dim: int = 128, backbone_name: str = "resnet34"):
        super().__init__()
        b_name = backbone_name.lower()
        if "resnet50" in b_name:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "resnet18" in b_name:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "small" in b_name:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif "large" in b_name or "mobile" in b_name:
            self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim)
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
