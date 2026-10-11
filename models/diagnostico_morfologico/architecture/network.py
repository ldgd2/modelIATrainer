import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)

def get_feature_backbone(backbone_name: str = "resnet34", pretrained: bool = True):
    name = backbone_name.lower()
    if "resnet50" in name:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif "resnet18" in name:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif "large" in name:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif "small" in name or "mobilenet" in name:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    else:  # default resnet34
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    return model, in_features

class ColorimetryNet(nn.Module):
    """Clasificador de colorimetria, tono y subtono cutaneo sobre GPU."""
    def __init__(self, num_classes: int = 4, backbone_name: str = "resnet34"):
        super().__init__()
        self.backbone, in_feat = get_feature_backbone(backbone_name, pretrained=True)
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

class MorphologyNet(nn.Module):
    """Estimador de proporciones corporales y keypoints COCO."""
    def __init__(self, num_outputs: int = 34, backbone_name: str = "resnet34"):
        super().__init__()
        self.backbone, in_feat = get_feature_backbone(backbone_name, pretrained=True)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

class HairSegmentationNet(nn.Module):
    """U-Net convolucional profunda con skip connections para segmentacion capilar."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 112x112

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 56x56

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 28x28

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder con Skip Connections
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        mask = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Redimensionar a tamano estandar si es necesario
        if mask.shape[-2:] != (64, 64):
            mask = nn.functional.interpolate(mask, size=(64, 64), mode="bilinear", align_corners=False)
        return mask

def build_diagnostico_model(task: str, backbone_name: str = "resnet34") -> nn.Module:
    if task == "colorimetria":
        return ColorimetryNet(num_classes=4, backbone_name=backbone_name)
    elif task == "morfologia":
        return MorphologyNet(num_outputs=34, backbone_name=backbone_name)
    elif task == "capilar":
        return HairSegmentationNet()
    else:
        raise ValueError(f"Tarea desconocida: {task}")
