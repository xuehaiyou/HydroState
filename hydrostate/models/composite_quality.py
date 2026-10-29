"""Encode ordered per-pixel observation support without altering spectral bands."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.data.retrieval.contract import INPUT_GROUPS, QUALITY_CHANNELS, TIME_STEPS
from hydrostate.registry import MODELS


@MODELS.register_module()
class CompositeQualityFusion(nn.Module):
    def __init__(self, feature_channels=768, hidden_channels=32):
        super().__init__()
        # Five slots, six native groups, four statistics plus availability.
        self.layers = nn.Sequential(
            nn.Conv2d(TIME_STEPS * len(INPUT_GROUPS) * (QUALITY_CHANNELS + 1), hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, 1, bias=False),
        )
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, features, inputs, *, cache=None):
        # All depth/task exits query the same input support. Keep its encoding
        # only in the caller's current-forward cache, including its gradient.
        size = tuple(features.shape[-2:])
        if cache is not None and size in cache:
            return features + cache[size]
        parts = []
        for name in INPUT_GROUPS:
            quality = inputs[f"{name}_quality"].float()
            valid = inputs[f"{name}_valid"].bool()
            if quality.shape[1:3] != (TIME_STEPS, QUALITY_CHANNELS):
                raise ValueError("Quality fusion requires five slots and four statistics")
            # Count is valid dates (at most eight); dates are offsets within slot.
            parts.append((quality / 8).masked_fill(~valid, 0).flatten(1, 2))
            parts.append(valid.float().flatten(1, 2))
        quality = torch.cat(parts, dim=1)
        encoded = self.layers(F.adaptive_avg_pool2d(quality, size))
        if cache is not None:
            cache[size] = encoded
        return features + encoded
