"""Small auxiliary encoders that preserve pretrained satellite input contracts."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS


class AuxiliaryBranch(nn.Module):
    def __init__(self, channels, feature_channels, mean, std, temporal="static",
                 hidden_channels=32):
        super().__init__()
        if temporal not in {"static", "mean"}:
            raise ValueError("Auxiliary temporal mode must be static or explicit masked mean")
        if min(channels, feature_channels, hidden_channels) <= 0:
            raise ValueError("Auxiliary channel counts must be positive")
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.numel() != channels or std.numel() != channels:
            raise ValueError("Supply one fixed normalization mean/std per auxiliary channel")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Auxiliary normalization must be finite with positive std")
        self.register_buffer("mean", mean.reshape(1, channels, 1, 1))
        self.register_buffer("std", std.reshape(1, channels, 1, 1))
        self.temporal = temporal
        self.channels = channels
        self.layers = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, 1), nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, 1, bias=False),
        )

    def forward(self, value, valid, size):
        value, valid = value.float(), valid.bool()
        expected_dims = 4 if self.temporal == "static" else 5
        if value.ndim != expected_dims or value.shape != valid.shape:
            raise ValueError("Auxiliary input/mask dimensions disagree with temporal mode")
        if value.shape[-3] != self.channels:
            raise ValueError("Auxiliary input channels disagree with configuration")
        value = value.masked_fill(~valid, 0)
        if self.temporal == "mean":
            count = valid.sum(dim=1)
            value = value.sum(dim=1) / count.clamp(min=1)
            valid = count > 0
        value = ((value - self.mean) / self.std).masked_fill(~valid, 0)
        # Pool values and masks together so absent pixels cannot dilute observations.
        fraction = F.adaptive_avg_pool2d(valid.float(), size)
        value = F.adaptive_avg_pool2d(value, size) / fraction.clamp(min=1e-6)
        encoded = self.layers(torch.cat((value, fraction), dim=1))
        return encoded.masked_fill(~(fraction > 0).any(dim=1, keepdim=True), 0)


@MODELS.register_module()
class AuxiliaryInputFusion(nn.Module):
    """Add masked auxiliary features to one or several satellite feature levels.

    Input values are in the pack's declared units. Mean/std must be fixed physical
    scales or training-only statistics. Sequence reduction is explicitly chosen;
    order-sensitive sequences require a different registered fusion module.
    """

    def __init__(self, inputs, feature_channels=768):
        super().__init__()
        if not inputs:
            raise ValueError("AuxiliaryInputFusion requires at least one input")
        self.feature_channels = feature_channels
        self.branches = nn.ModuleDict({
            name: AuxiliaryBranch(feature_channels=feature_channels, **spec)
            for name, spec in inputs.items()
        })

    def forward(self, features, inputs):
        multiple = isinstance(features, (list, tuple))
        levels = features if multiple else [features]
        outputs = []
        for feature in levels:
            if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
                raise ValueError("Auxiliary fusion feature channels disagree with encoder")
            output = feature
            for name, branch in self.branches.items():
                output = output + branch(inputs[name], inputs[f"{name}_valid"], feature.shape[-2:])
            outputs.append(output)
        return tuple(outputs) if multiple else outputs[0]
