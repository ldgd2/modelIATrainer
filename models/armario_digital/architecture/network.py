import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ArmarioClassifier(nn.Module):
    """
    Lightweight, high-accuracy backbone for garment classification.
    Ideal for fast training iterations on Google Colab GPU/T4.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Factory function for model creation."""
    return ArmarioClassifier(num_classes=num_classes, pretrained=pretrained)
