import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ColorimetriaClassifier(nn.Module):
    """
    ResNet-based architecture for facial seasonal color analysis (BellAI Colorimetria).
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    return ColorimetriaClassifier(num_classes=num_classes, pretrained=pretrained)
