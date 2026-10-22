"""Encode ordered per-pixel observation support without altering spectral bands."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.data.retrieval.contract import QUALITY_CHANNELS, SENSORS, TIME_STEPS
from hydrostate.registry import MODELS


@MODELS.register_module()
class CompositeQualityFusion(nn.Module):
    def __init__(self, feature_channels=768, hidden_channels=32):
        super().__init__()
        # Five slots, three sensors, four statistics plus explicit availability.
        self.layers = nn.Sequential(
            nn.Conv2d(TIME_STEPS * len(SENSORS) * (QUALITY_CHANNELS + 1), hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, 1, bias=False),
        )
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, features, inputs):
        parts = []
        for sensor in SENSORS:
            quality = inputs[f"{sensor}_quality"].float()
            valid = inputs[f"{sensor}_valid"].bool()
            if quality.shape[1:3] != (TIME_STEPS, QUALITY_CHANNELS):
                raise ValueError("Quality fusion requires five slots and four statistics")
            # Count is valid dates (at most eight); dates are offsets within slot.
            parts.append((quality / 8).masked_fill(~valid, 0).flatten(1, 2))
            parts.append(valid.float().flatten(1, 2))
        quality = torch.cat(parts, dim=1)
        return features + self.layers(F.adaptive_avg_pool2d(quality, features.shape[-2:]))
