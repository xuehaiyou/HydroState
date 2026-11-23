"""DiTASK spectral projections using the authors' boundary-fixed difw CPAB backend."""

import math
from contextlib import nullcontext
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..mtlora import TaskExitBlock
from .state import CoSAStateFusion


class AdapterDropout(nn.Dropout):
    """Trainable-path dropout controlled independently of frozen backbone stochasticity."""


@lru_cache(maxsize=16)
def cpab(cells, device):
    from difw import Cpab

    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        return Cpab(cells, "pytorch", "gpu" if device.type == "cuda" else "cpu",
                    zero_boundary=True)


def coefficients(cells):
    value = nn.Parameter(torch.empty(1, cells - 1))
    nn.init.kaiming_uniform_(value, a=math.sqrt(5))
    return value


class SpectralProjection(nn.Module):
    def __init__(self, base, tasks=(), shared_theta=None, drop_shared=True):
        super().__init__()
        # Frozen W: cache its economy SVD once; it is part of the checkpoint.
        u, s, vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
        self.register_buffer("u", u.contiguous())
        self.register_buffer("s", s.contiguous())
        self.register_buffer("vh", vh.contiguous())
        self.bias = base.bias
        self.shared = coefficients(64) if shared_theta is None else shared_theta
        self.private = nn.ParameterDict({task: coefficients(4) for task in tasks})
        self.dropout = AdapterDropout(0.05) if drop_shared else nn.Identity()

    def forward(self, x, task=None):
        theta = self.shared if task is None else self.private[task]
        device = theta.device
        # difw uses the current CUDA device internally to construct its sampling grid.
        with (torch.cuda.device(device) if device.type == "cuda" else nullcontext(),
              torch.autocast(device.type, enabled=False)):
            flow = cpab(theta.shape[-1] + 1, device)
            warped = flow.transform_data(self.s.view(1, -1, 1), theta,
                                         outsize=self.s.numel()).flatten()
            weight = (self.u * warped) @ self.vh
        # Official FC task paths consume the already task-specific activations.
        return F.linear(self.dropout(x) if task is None else x, weight, self.bias)


def adapt_spectral_block(block, stage, tasks):
    # The official fused QKV uses one deformation for three independently sliced
    # spectra and one input dropout mask. OlmoEarth exposes separate Q/K/V maps.
    shared_qkv = coefficients(64)
    for name in ("q", "k", "v"):
        base = getattr(block.attn, name)
        setattr(block.attn, name, SpectralProjection(base, shared_theta=shared_qkv,
                                                     drop_shared=False))
    block.norm1 = nn.Sequential(block.norm1, AdapterDropout(0.05))
    if stage:
        block = TaskExitBlock(block, tasks, True, lambda base: SpectralProjection(base, tasks))
        # Shared attention result is dropped once before projection for all branches.
        block.attn.proj = AdapterDropout(0.05)
        block.projection.dropout = nn.Identity()
    else:
        block.attn.proj = SpectralProjection(block.attn.proj)
        block.mlp.fc1 = SpectralProjection(block.mlp.fc1)
        block.mlp.fc2 = SpectralProjection(block.mlp.fc2)
    return block


@MODELS.register_module()
class DiTASKStateFusion(CoSAStateFusion):
    """Task-specific four-depth projections and HRNet readout on the common OlmoEarth grid."""

    def _build_depth_fusion(self, num_levels, channels):
        widths = (18, 36, 72, 144)
        if num_levels != len(widths):
            raise ValueError("DiTASK expects four stage outputs")
        self.adapters = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        self.head_channels = 4 * sum(widths)
        for task in ("coverage", "sm", "et"):
            self.adapters[task] = nn.ModuleList(nn.Conv2d(channels, w, 1, bias=False)
                                                for w in widths)
            self.fusion[task] = nn.Sequential(nn.Conv2d(sum(widths), self.head_channels, 1),
                                              nn.BatchNorm2d(self.head_channels), nn.ReLU())

    def forward(self, prepared, bounds, *, task):
        _, quality, valid = prepared
        levels = self.pool_levels(prepared, bounds)
        projected = [layer(x + quality) for layer, x in zip(self.adapters[task], levels)]
        return self.fusion[task](torch.cat(projected, dim=1)).masked_fill(~valid, 0), valid
