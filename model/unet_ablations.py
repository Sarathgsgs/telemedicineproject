"""
model/unet_ablations.py
-----------------------
Ablation variants of the Attention U-Net architecture for Phase 10:
1. Normalization Ablation: GroupNorm (Ours) vs BatchNorm2d (Standard).
2. Attention Ablation: Attention Gates (Ours) vs Direct Skip Connections (Vanilla U-Net).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from model.unet_attention import AttentionGate


class ConvBlockAblation(nn.Module):
    """
    Configurable ConvBlock supporting either GroupNorm or BatchNorm2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group_norm",
        num_groups: int = 8
    ):
        super().__init__()
        self.norm_type = norm_type.lower()

        if self.norm_type == "batch_norm":
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)
        else:
            # GroupNorm
            groups = min(num_groups, out_channels)
            while out_channels % groups != 0:
                groups //= 2
            norm1 = nn.GroupNorm(groups, out_channels)
            norm2 = nn.GroupNorm(groups, out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetAblation(nn.Module):
    """
    Ablation-configurable U-Net:
    - norm_type: 'group_norm' | 'batch_norm'
    - use_attention: True (Attention Gates) | False (Vanilla Skip Connections)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 32,
        norm_type: str = "group_norm",
        use_attention: bool = True
    ):
        super().__init__()
        self.norm_type = norm_type
        self.use_attention = use_attention
        f = [base_filters * (2**i) for i in range(5)]  # [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlockAblation(in_channels, f[0], norm_type=norm_type)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlockAblation(f[0], f[1], norm_type=norm_type)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlockAblation(f[1], f[2], norm_type=norm_type)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlockAblation(f[2], f[3], norm_type=norm_type)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlockAblation(f[3], f[4], norm_type=norm_type)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        if use_attention:
            self.ag4 = AttentionGate(f_g=f[3], f_l=f[3], f_int=f[2])
        self.dec4 = ConvBlockAblation(f[4], f[3], norm_type=norm_type)

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        if use_attention:
            self.ag3 = AttentionGate(f_g=f[2], f_l=f[2], f_int=f[1])
        self.dec3 = ConvBlockAblation(f[3], f[2], norm_type=norm_type)

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        if use_attention:
            self.ag2 = AttentionGate(f_g=f[1], f_l=f[1], f_int=f[0])
        self.dec2 = ConvBlockAblation(f[2], f[1], norm_type=norm_type)

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        if use_attention:
            self.ag1 = AttentionGate(f_g=f[0], f_l=f[0], f_int=f[0] // 2)
        self.dec1 = ConvBlockAblation(f[1], f[0], norm_type=norm_type)

        self.final_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        x4 = self.ag4(g=d4, x=e4)[0] if self.use_attention else e4
        d4 = self.dec4(torch.cat([x4, d4], dim=1))

        d3 = self.up3(d4)
        x3 = self.ag3(g=d3, x=e3)[0] if self.use_attention else e3
        d3 = self.dec3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        x2 = self.ag2(g=d2, x=e2)[0] if self.use_attention else e2
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        x1 = self.ag1(g=d1, x=e1)[0] if self.use_attention else e1
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        return self.final_conv(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
