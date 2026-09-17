"""
model/unet_attention.py
-----------------------
2D Attention U-Net-Lite with Group Normalization for Federated Medical Imaging.
Replaces BatchNorm with GroupNorm to eliminate batch-size instability in small,
heterogeneous local client batches. Integrates soft Attention Gates at skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """
    Dual 3x3 Conv block with GroupNorm and LeakyReLU.
    GroupNorm computes statistics per-sample across channel groups,
    making it completely invariant to batch size in federated learning.
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides out_channels
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0:
            groups //= 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft Attention Gate (Oktay et al., MIDL 2018).
    Suppresses activations in non-target abdominal tissue (kidneys, spleen, bowel)
    and highlights salient liver parenchyma and focal lesion boundaries.
    
    Args:
        f_g: Number of feature channels from gating signal (coarser/deeper scale).
        f_l: Number of feature channels from skip connection (finer/shallower scale).
        f_int: Intermediate channel dimension.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        groups = min(8, f_int)
        while f_int % groups != 0:
            groups //= 2

        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: Gating signal from decoder (B, f_g, H_g, W_g)
            x: Spatial feature from encoder skip (B, f_l, H_x, W_x)
        Returns:
            (attenuated_x, attention_coefficients)
        """
        # Upsample gating signal if spatial dims differ
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # Attention map in [0, 1]

        return x * alpha, alpha


class AttentionUNetLite(nn.Module):
    """
    Lightweight 2D Attention U-Net with GroupNorm.
    Parameter count: ~1.8M (vs 31.2M in 3D base paper), optimized for laptop/Colab FL.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_filters: int = 32):
        super().__init__()
        f = [base_filters * (2**i) for i in range(5)]  # [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f[0], f[1])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f[1], f[2])
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(f[2], f[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f[3], f[4])

        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.ag4 = AttentionGate(f_g=f[3], f_l=f[3], f_int=f[2])
        self.dec4 = ConvBlock(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.ag3 = AttentionGate(f_g=f[2], f_l=f[2], f_int=f[1])
        self.dec3 = ConvBlock(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.ag2 = AttentionGate(f_g=f[1], f_l=f[1], f_int=f[0])
        self.dec2 = ConvBlock(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.ag1 = AttentionGate(f_g=f[0], f_l=f[0], f_int=f[0] // 2)
        self.dec1 = ConvBlock(f[1], f[0])

        # Final Classifier Head
        self.final_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False
    ) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder + Attention
        d4 = self.up4(b)
        x4, a4 = self.ag4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([x4, d4], dim=1))

        d3 = self.up3(d4)
        x3, a3 = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        x2, a2 = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        x1, a1 = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        logits = self.final_conv(d1)

        if return_attention_maps:
            return logits, [a1, a2, a3, a4]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
