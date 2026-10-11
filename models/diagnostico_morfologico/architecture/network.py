import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ColorimetryNet(nn.Module):
    """Clasificador de colorimetria, tono y subtono cutaneo."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class MorphologyNet(nn.Module):
    """Estimador de proporciones corporales y keypoints COCO."""
    def __init__(self, num_outputs: int = 34):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feat, num_outputs)
        )

    def forward(self, x):
        return self.backbone(x)

class HairSegmentationNet(nn.Module):
    """Segmentador convolucional de masa y limites capilares."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 56x56
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        mask = self.decoder(feat)
        return mask

def build_diagnostico_model(task: str) -> nn.Module:
    if task == "colorimetria":
        return ColorimetryNet(num_classes=4)
    elif task == "morfologia":
        return MorphologyNet(num_outputs=34)
    elif task == "capilar":
        return HairSegmentationNet()
    else:
        raise ValueError(f"Tarea desconocida: {task}")
