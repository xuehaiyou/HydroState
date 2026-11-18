"""Complete four-depth MLoRE decoder with the official ViT-B/PASCAL settings.

Only the encoder/temporal interface, three output groups and final output grid
are HydroState-specific. No CoSA private expert is inserted before MLoRE.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..composite_quality import CompositeQualityFusion
from ..cosa.state import TargetStatePooler
from .vendor import MOEBlock


class IntervalDepthPool(nn.Module):
    """Keep encoder depth separate while querying the same target interval."""

    def __init__(self, channels, num_heads=4, quality_channels=32):
        super().__init__()
        self.channels = channels
        self.pooler = TargetStatePooler(channels, num_heads)
        self.quality = CompositeQualityFusion(channels, quality_channels)

    def prepare(self, levels, inputs):
        if len(levels) != 4:
            raise ValueError('Complete MLoRE requires four encoder depths')
        shape = levels[0]['features'].shape
        if len(shape) != 5 or shape[2] != self.channels:
            raise ValueError('Expected B,T,C,H,W encoder levels')
        prepared = []
        for level in levels:
            value, valid, times = level['features'], level['valid'], level['time']
            if (value.shape != shape or valid.shape != (shape[0], shape[1], *shape[-2:])
                    or times.shape != shape[:2] or valid.dtype != torch.bool):
                raise ValueError('Depths must share grid, timestamps and observation masks')
            prepared.append((value.masked_fill(~valid[:, :, None], 0).permute(0, 1, 3, 4, 2),
                             valid, times))
        valid = torch.stack([mask.any(1) for _, mask, _ in prepared]).any(0)[:, None]
        reference = levels[0]['features'][:, 0]
        quality = self.quality(torch.zeros_like(reference), inputs).masked_fill(~valid, 0)
        return prepared, quality, valid

    def forward(self, prepared, bounds):
        levels, quality, valid = prepared
        pooled = [self.pooler(value, mask, times, bounds, quality)
                  for value, mask, times in levels]
        return pooled, valid


class FourDepthMLoRE(nn.Module):
    """Official two-block-per-depth computation and task-specific depth masks."""

    def __init__(self, groups, in_channels=768, channels=640, grid_size=30,
                 ranks=tuple(range(16, 129, 8)), private_rank=64, topk=9,
                 pre_softmax=False):
        super().__init__()
        self.groups = tuple(groups)
        if (not self.groups or len(set(self.groups)) != len(self.groups)
                or channels < 4 or channels % 4 or grid_size < 1 or in_channels < 1
                or len(ranks) < 2 or min(ranks) < 1 or private_rank < 1
                or not 1 <= topk <= len(ranks)):
            raise ValueError('Invalid complete MLoRE configuration')
        self.grid_size = grid_size
        options = SimpleNamespace(TASKS=SimpleNamespace(NAMES=self.groups),
                                  rank_list=list(ranks), spe_rank=private_rank,
                                  topk=topk, pre_softmax=pre_softmax)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.fea_fuse = nn.ModuleList([nn.Conv2d(in_channels, channels, 1) for _ in range(4)])
        self.MLoRE_1 = nn.ModuleList([
            MOEBlock(options, channels, grid_size**2, kernel_size=3) for _ in range(4)])
        self.MLoRE_2 = nn.ModuleList([
            MOEBlock(options, channels, grid_size**2, kernel_size=3) for _ in range(4)])
        self.task_mask = nn.ModuleDict({task: nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1), nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, 4, 3, padding=1)) for task in self.groups})
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        # Matches the active initialization in official _init_vit_weights.
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, levels, valid):
        if len(levels) != 4 or any(x.shape[-2:] != (self.grid_size, self.grid_size)
                                  for x in levels):
            raise ValueError('Four matching configured feature grids are required')
        out = {task: [] for task in self.groups}
        routes = []
        for index, value in enumerate(levels):
            if index == 3:
                value = self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            value = self.fea_fuse[index](value.masked_fill(~valid, 0))
            first_routes, second_routes = {}, {}
            for task in self.groups:
                first, _, first_routes[task] = self.MLoRE_1[index](value, task)
                second, _, second_routes[task] = self.MLoRE_2[index](first, task)
                out[task].append(second)
            routes.extend((first_routes, second_routes))
        fused = {}
        for task in self.groups:
            weights = self.task_mask[task](torch.cat(out[task], dim=1)).softmax(1)
            value = sum(weights[:, i:i + 1] * out[task][i] for i in range(4))
            fused[task] = F.interpolate(value.masked_fill(~valid, 0), scale_factor=4,
                                       mode='bilinear', align_corners=False)
        return fused, routes


def load_balance_loss(routes, eligible):
    """Official sum of squared CV across all tasks, for each of eight blocks.

    Entirely unobserved windows are excluded; all-valid batches reproduce the
    published formula, including sample-standard-deviation correction.
    """
    total = next(iter(routes[0].values())).sum() * 0
    if not eligible.any():
        return total
    for layer in routes:
        probabilities = torch.cat([value[eligible].float() for value in layer.values()], dim=0)
        importance = probabilities.mean(0).reshape(-1)
        total = total + (importance.std() / importance.mean().clamp_min(1e-12)).square()
    return total


@MODELS.register_module()
class CompleteMLoREDecoder(nn.Module):
    def __init__(self, groups, in_channels=768, channels=640, grid_size=30,
                 ranks=tuple(range(16, 129, 8)), private_rank=64, topk=9,
                 pre_softmax=False, balance_weight=.0001, quality_channels=32):
        super().__init__()
        if balance_weight < 0:
            raise ValueError('Negative expert balance coefficient')
        self.groups, self.out_channels = tuple(groups), channels
        self.temporal = IntervalDepthPool(in_channels, quality_channels=quality_channels)
        self.network = FourDepthMLoRE(self.groups, in_channels, channels, grid_size,
                                      ranks, private_rank, topk, pre_softmax)
        self.balance_weight = balance_weight

    def prepare(self, levels, inputs):
        return self.temporal.prepare(levels, inputs)

    def forward(self, prepared, bounds):
        levels, valid = self.temporal(prepared, bounds)
        features, routes = self.network(levels, valid)
        auxiliary = (self.balance_weight * load_balance_loss(routes, valid.flatten(1).any(1))
                     if self.training else None)
        valid = F.interpolate(valid.float(), scale_factor=4, mode='nearest').bool()
        return features, valid, auxiliary
