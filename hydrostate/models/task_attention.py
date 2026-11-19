"""Four interval-aligned OlmoEarth exits for task attention experiments."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS
from .mlore.decoder import IntervalDepthPool


class PromptFilter(nn.Module):
    """Task prompt spatial attention followed by a rank-eight dynamic task filter."""
    def __init__(self, c, rank=8):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, c, 1, 1) / c**.5)
        self.down = nn.Conv2d(c, rank, 1)
        self.kernel = nn.Conv2d(rank, rank * 9, 1)
        self.scale = nn.Parameter(torch.randn(1, rank, 9) * (2**.5 / 3))
        self.shift = nn.Parameter(torch.zeros(1, rank, 9))
        self.up = nn.Conv2d(rank, c, 1)

    def forward(self, x, valid):
        scores = (x * self.prompt).sum(1, keepdim=True).float().masked_fill(~valid, -1e30)
        attention = scores.flatten(2).softmax(-1).reshape_as(scores).to(x.dtype)
        x = x * (1 + attention)
        z = self.down(x).masked_fill(~valid, 0)
        b, r, h, w = z.shape
        pooled = z.sum((-2, -1), keepdim=True) / valid.sum((-2, -1), keepdim=True).clamp_min(1)
        kernels = self.kernel(pooled).reshape(b, r, 9).float()
        kernels = (kernels - kernels.mean(-1, keepdim=True)) / kernels.std(-1, keepdim=True).clamp_min(1e-6)
        kernels = (kernels * self.scale + self.shift).reshape(b*r, 1, 3, 3)
        filtered = F.conv2d(z.reshape(1, b*r, h, w), kernels, padding=1, groups=b*r)
        return x + self.up(filtered.reshape_as(z))


@MODELS.register_module()
class TaskAttentionDecoder(nn.Module):
    def __init__(self, method, groups, in_channels=768, channels=64):
        super().__init__()
        self.groups, self.method, self.out_channels = tuple(groups), method, channels
        self.temporal = IntervalDepthPool(in_channels)
        self.shared = nn.ModuleList(nn.Conv2d(in_channels, channels, 1) for _ in range(4))
        if method == 'bridge':
            from .bridgenet.transformer_decoder import TransformerDecoder
            options = SimpleNamespace(backbone='vitB', backbone_channels=[channels]*4,
                embed_dim=64, PRED_OUT_NUM_CONSTANT=32, spatial_dim=[(16, 16)]*4,
                mtt_resolution_downsample_rate=2, REFINE_DEPTH=[12]*3,
                DILATED_RATIO=[[1, 2, 5]*4]*3, SQUEEZE_RATIO=[[2]*12]*3,
                TASKS=SimpleNamespace(NAMES=self.groups, NUM_OUTPUT=dict(coverage=3, sm=1, et=3)))
            self.bridge = TransformerDecoder(options)
            self.output = nn.ModuleDict({t: nn.Conv2d(96, channels, 1) for t in groups})
        elif method in ('mtan', 'tad'):
            self.layers = nn.ModuleList(nn.ModuleDict({t: (
                nn.Sequential(nn.Conv2d(channels*(1 if i == 0 else 2), channels, 1),
                              nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 1),
                              nn.Sigmoid()) if method == 'mtan' else PromptFilter(channels))
                for t in groups}) for i in range(4))
            self.transitions = nn.ModuleList(nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU()) for _ in range(3))
        else:
            raise ValueError(method)

    def prepare(self, levels, inputs):
        return self.temporal.prepare(levels, inputs)

    def forward(self, prepared, bounds):
        levels, valid = self.temporal(prepared, bounds)
        levels = [proj(x).masked_fill(~valid, 0) for proj, x in zip(self.shared, levels)]
        if self.method == 'bridge':
            features, _ = self.bridge([F.interpolate(x, size=(16, 16), mode='bilinear',
                                                      align_corners=False) for x in levels])
            features = {t: self.output[t](x) for t, x in features.items()}
            valid = F.interpolate(valid.float(), size=next(iter(features.values())).shape[-2:],
                                  mode='nearest').bool()
        else:
            features = {}
            for i, x in enumerate(levels):
                for t in self.groups:
                    previous = self.transitions[i-1](features[t]).masked_fill(~valid, 0) if i else None
                    if self.method == 'mtan':
                        z = x if previous is None else torch.cat([x, previous], 1)
                        y = self.layers[i][t](z) * x
                    else:
                        y = self.layers[i][t](x if previous is None else x + previous, valid)
                    features[t] = y.masked_fill(~valid, 0)
        return features, valid, None
