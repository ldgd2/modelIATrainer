import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Bloque residual convolucional con Instance Normalization."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class TryOnGeneratorNet(nn.Module):
    """
    Generador profundo U-Net con bloques residuales para probador virtual (VITON-HD).
    Entrada: 6 canales (imagen corporal + imagen de prenda).
    Salida: 3 canales (resultado vestido sintetizado).
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        # Encoder (Downsampling)
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck residual (Alta capacidad para GPU)
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # Decoder (Upsampling con Skip Connections)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.bottleneck(e4)

        d4 = self.d4(b)
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        out = self.d1(torch.cat([d2, e1], dim=1))
        return out

class MakeupTransferNet(nn.Module):
    """
    Red generativa residual profunda para mapeo de maquillaje (MT-Dataset).
    Aplica 6 bloques residuales de transformacion de textura y tono facial.
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 6 Bloques Residuales para modelado de sombras y maquillaje
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(6)])
        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.initial(x)
        feat = self.down(feat)
        feat = self.res_blocks(feat)
        return self.up(feat)

def build_simulacion_model(task: str) -> nn.Module:
    if task == "maquillaje":
        return MakeupTransferNet()
    return TryOnGeneratorNet()
