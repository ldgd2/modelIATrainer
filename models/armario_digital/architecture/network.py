import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights
)

class ArmarioClassifier(nn.Module):
    """
    Backbone configurable para clasificacion y segmentacion de prendas.
    Soporta MobileNet (ultraligero) y ResNet (alto rendimiento para GPU).
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True, backbone_name: str = "resnet34"):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if "resnet50" in self.backbone_name:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
        elif "resnet34" in self.backbone_name or "resnet" in self.backbone_name:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
        elif "large" in self.backbone_name:
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_classes)
            )
        else:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def build_model(num_classes: int = 10, pretrained: bool = True, backbone_name: str = "resnet34") -> nn.Module:
    """Factory function for model creation."""
    return ArmarioClassifier(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
