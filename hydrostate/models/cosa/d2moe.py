"""Task-adaptive attention and semantic prior injection from D2MoE (Eqs. 1–22)."""

import math
from contextvars import ContextVar

import torch
from torch import nn

from hydrostate.registry import MODELS

from ..shared_private_lora import LowRankUpdate
from .ditask import AdapterDropout
from .state import CoSAStateFusion

_PRIOR = object()
_ROUTE = ContextVar("d2moe_route", default=None)


class MixtureProjection(nn.Module):
    def __init__(self, base, experts=0):
        super().__init__()
        self.base = base
        self.shared = LowRankUpdate(base.in_features, base.out_features, 16, scale=4 / 16)
        self.dropout = AdapterDropout(0.05)
        if experts:
            rank = 4
            self.scale = 4 / rank
            self.down = nn.Parameter(torch.empty(experts, rank, base.in_features))
            self.up = nn.Parameter(torch.zeros(experts, base.out_features, rank))
            for a in self.down.unbind():
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x):
        route = _ROUTE.get()
        base = self.base(x)
        if route is _PRIOR:  # Unmodified stage block for the frozen semantic prior.
            return base
        inputs = self.dropout(x)
        if route is None:
            return base + self.shared(inputs)
        # Eq. 11: the task exit uses the frozen projection plus the selected
        # expert update. Shared updates reach this exit through previous blocks.
        indices, probabilities = route
        hidden = torch.einsum("bni,bkri->bknr", inputs, self.down[indices])
        hidden = hidden * probabilities[:, :, None, None].to(hidden.dtype)
        delta = torch.einsum("bknr,bkor->bno", hidden, self.up[indices])
        return base + self.scale * delta


class MixtureExit(nn.Module):
    def __init__(self, block, groups, experts):
        super().__init__()
        self.block = block
        self.gates = nn.ModuleDict({name: nn.Linear(block.norm1.weight.numel(), experts, bias=False)
                                    for name in groups})

    def forward(self, x, **kwargs):
        shared = self.block(x=x, **kwargs)
        valid = kwargs["attn_mask"]
        summary = x.masked_fill(~valid[..., None], 0).float().sum(1)
        summary = (summary / valid.sum(1, keepdim=True).clamp_min(1)).to(x.dtype)
        branches = []
        for gate in self.gates.values():
            scores, indices = gate(summary).topk(8)
            ctx = _ROUTE.set((indices, scores.float().softmax(-1)))
            try:
                branches.append(self.block(x=x, **kwargs))
            finally:
                _ROUTE.reset(ctx)
        ctx = _ROUTE.set(_PRIOR)
        try:
            with torch.no_grad():
                prior = self.block(x=x, **kwargs)
        finally:
            _ROUTE.reset(ctx)
        return shared, *branches, prior


def adapt_mixture_block(block, stage, groups):
    for name in ("q", "k", "v", "proj"):
        base = getattr(block.attn, name)
        setattr(block.attn, name, MixtureProjection(base, experts=16 if stage else 0))
    return MixtureExit(block, groups, 16) if stage else block


class SemanticPriorInjector(nn.Module):
    """Fuse interval-aligned maps on the native feature grid; input shape D,B,C,H,W."""

    def __init__(self, channels, depth, groups, width=384):
        super().__init__()
        self.align = nn.Conv2d(channels, width, 1)
        self.align_prior = nn.Conv2d(channels, width, 1)
        self.logits = nn.ModuleDict({group: nn.Sequential(
            nn.Conv2d(depth * width, width, 1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, depth, 3, padding=1)) for group in groups})
        self.prior_fusion = nn.Sequential(nn.Conv2d(depth * width, width, 1),
                                          nn.Conv2d(width, width, 3, padding=1))
        self.refine = nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.BatchNorm2d(width),
                                    nn.GELU(), nn.Conv2d(width, channels, 3, padding=1))

    @staticmethod
    def aligned(levels, projection, valid):
        depth, batch = levels.shape[:2]
        maps = projection(levels.flatten(0, 1)).reshape(depth, batch, -1, *levels.shape[-2:])
        return maps.transpose(0, 1).masked_fill(~valid[:, None], 0)

    def merge_prior(self, levels, valid):
        maps = self.aligned(levels, self.align_prior, valid)
        return self.prior_fusion(maps.flatten(1, 2)).masked_fill(~valid, 0)

    def forward(self, group, levels, prior, valid):
        maps = self.aligned(levels, self.align, valid)
        weights = self.logits[group](maps.flatten(1, 2)).float().softmax(1)
        fused = (maps * weights[:, :, None].to(maps.dtype)).sum(1)
        return self.refine(torch.cat((fused, prior), dim=1)).masked_fill(~valid, 0)


@MODELS.register_module()
class D2MoEStateFusion(CoSAStateFusion):
    """Pool each depth at the requested time, then apply SPI on the common grid."""

    def _build_depth_fusion(self, num_levels, channels):
        self.head_channels = channels
        self.spi = SemanticPriorInjector(channels, num_levels, ("coverage", "sm", "et"),
                                         width=channels)

    def prior_map(self, prepared, bounds):
        return self.spi.merge_prior(self.pool_levels(prepared, bounds), prepared[2])

    def forward(self, prepared, bounds, *, task, prior):
        _, quality, valid = prepared
        levels = self.pool_levels(prepared, bounds)
        return self.spi(task, levels, prior, valid) + quality, valid
