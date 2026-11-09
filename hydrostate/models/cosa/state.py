"""Shared interval-conditioned temporal pooling and same-grid depth fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..composite_quality import CompositeQualityFusion
from ..query_time import query_time_features
from .branches import SpatialResidual


class TargetStatePooler(nn.Module):
    """Four local temporal attention heads; no spatial all-pairs attention."""

    def __init__(self, channels=256, num_heads=4):
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError("State channels must be positive and divisible by attention heads")
        self.channels, self.num_heads = channels, num_heads
        self.time_query = nn.Sequential(nn.Linear(5, channels), nn.GELU(),
                                        nn.Linear(channels, channels))
        self.relative_score = nn.Sequential(nn.Linear(3, channels), nn.GELU(),
                                            nn.Linear(channels, num_heads))
        self.quality_query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, value, valid, times, bounds, quality):
        # value B,T,H,W,C; valid B,T,H,W; quality B,C,H,W.
        b, t, h, w, c = value.shape
        absolute, relative = query_time_features(bounds, times.to(value.device), b)
        query = (self.time_query(absolute)[:, None, None]
                 + self.quality_query(quality.permute(0, 2, 3, 1)))
        width = c // self.num_heads
        masked = value.masked_fill(~valid[..., None], 0)
        key = self.key(masked).reshape(b, t, h, w, self.num_heads, width)
        projected = self.value(masked).reshape(b, t, h, w, self.num_heads, width)
        query_heads = query.reshape(b, h, w, self.num_heads, width)
        scores = (key.float() * query_heads[:, None].float()).sum(-1) * width ** -0.5
        scores = scores + self.relative_score(relative)[:, :, None, None].float()
        scores = scores.masked_fill(~valid[..., None], -torch.finfo(scores.dtype).max)
        weights = scores.softmax(dim=1) * valid[..., None]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (weights.to(projected.dtype)[..., None] * projected).sum(1).reshape(b, h, w, c)
        # Retain interval conditioning even with only one usable observation.
        pooled = self.output(pooled) + query
        return pooled.permute(0, 3, 1, 2).masked_fill(~valid.any(1)[:, None], 0)


@MODELS.register_module()
class CoSAStateFusion(nn.Module):
    def __init__(self, in_channels=768, channels=256, num_levels=3, num_heads=4,
                 quality_channels=32):
        super().__init__()
        if min(in_channels, channels, num_levels, quality_channels) < 1:
            raise ValueError("State fusion dimensions must be positive")
        self.in_channels, self.out_channels = in_channels, channels
        self.projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, channels))
            for _ in range(num_levels))
        self.pooler = TargetStatePooler(channels, num_heads)
        self.quality = CompositeQualityFusion(channels, quality_channels)
        self.compress = nn.Conv2d(num_levels * channels, channels, 1)
        self.spatial = SpatialResidual(channels)

    def prepare(self, levels, inputs):
        """Project each sequence once; cache only within the current model forward."""
        if len(levels) != len(self.projections):
            raise ValueError("CoSA state fusion received the wrong number of depths")
        shape = levels[0]["features"].shape
        if len(shape) != 5 or shape[2] != self.in_channels:
            raise ValueError("Expected B,T,C,H,W encoder features")
        prepared = []
        for projection, level in zip(self.projections, levels):
            features, valid, times = level["features"], level["valid"], level["time"]
            if (features.shape != shape or valid.shape != (shape[0], shape[1], *shape[-2:])
                    or times.shape != shape[:2] or valid.dtype != torch.bool):
                raise ValueError("CoSA depths must share a grid and explicit observation support")
            features = features.masked_fill(~valid[:, :, None], 0).permute(0, 1, 3, 4, 2)
            prepared.append((projection(features).masked_fill(~valid[..., None], 0), valid, times))
        reference = prepared[0][0][:, 0].permute(0, 3, 1, 2)
        quality = self.quality(torch.zeros_like(reference), inputs)
        available = torch.stack([valid.any(1) for _, valid, _ in prepared]).any(0)[:, None]
        return prepared, quality.masked_fill(~available, 0), available

    def forward(self, prepared, bounds):
        levels, quality, valid = prepared
        pooled = [self.pooler(value, mask, times, bounds, quality)
                  for value, mask, times in levels]
        value = self.compress(torch.cat(pooled, dim=1)) + quality
        return self.spatial(F.gelu(value), valid), valid
