"""Task-specific regression heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS


def activate_prediction(output, activation):
    if activation == "sigmoid":
        return torch.sigmoid(output)
    if activation == "softplus":
        return F.softplus(output)
    if activation != "identity":
        raise ValueError(f"unsupported activation {activation!r}")
    return output


@MODELS.register_module()
class RegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        activation: str = "identity",
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        self.activation = activation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return activate_prediction(self.layers(features), self.activation)
