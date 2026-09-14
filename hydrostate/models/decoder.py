"""Shared spatial decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


@MODELS.register_module()
class SharedUNetDecoder(nn.Module):
    """Upsample a single OlmoEarth feature map to input resolution."""

    def __init__(
        self,
        in_channels: int = 768,
        channels: tuple[int, ...] | list[int] = (512, 256, 128),
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        if scale_factor < 1 or scale_factor & (scale_factor - 1):
            raise ValueError("scale_factor must be a positive power of two")
        self.scale_factor = scale_factor
        self.stem = ConvBlock(in_channels, channels[0])
        num_upsamples = scale_factor.bit_length() - 1
        if len(channels) != num_upsamples + 1:
            raise ValueError("channels must contain one stem and one entry per 2x upsample")
        blocks = []
        for left, right in zip(channels[:-1], channels[1:]):
            blocks.append(ConvBlock(left, right))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        return x

