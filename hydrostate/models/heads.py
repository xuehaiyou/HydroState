"""Task-specific regression heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS


@MODELS.register_module()
class RegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        activation: str = "identity",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.activation = activation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.layers(features)
        if self.activation == "sigmoid":
            return torch.sigmoid(output)
        if self.activation == "softplus":
            return F.softplus(output)
        if self.activation != "identity":
            raise ValueError(f"unsupported activation {self.activation!r}")
        return output
