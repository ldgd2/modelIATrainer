import torch
import torch.nn as nn

class TryOnGeneratorNet(nn.Module):
    """
    Generador convolucional U-Net para transferencia y ajuste de prenda sobre cuerpo (VITON-HD).
    Entrada: 6 canales (imagen corporal + imagen de prenda).
    Salida: 3 canales (resultado vestido sintetizado).
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        d3 = self.decoder3(e3)
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        out = self.decoder1(torch.cat([d2, e1], dim=1))
        return out

class MakeupTransferNet(nn.Module):
    """
    Red generativa residual para mapeo y transferencia de estilo de maquillaje (MT-Dataset).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def build_simulacion_model(task: str) -> nn.Module:
    if task == "probador":
        return TryOnGeneratorNet()
    elif task == "maquillaje":
        return MakeupTransferNet()
    else:
        return TryOnGeneratorNet()
