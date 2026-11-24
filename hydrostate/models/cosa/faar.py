"""Frequency-aware automatic-rank tuning (Fontana et al., CVPR 2026)."""

import math

import torch
from mmengine.hooks import Hook
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import HOOKS, MODELS

from ..mtlora import TaskExitBlock
from .ditask import DiTASKStateFusion


class RankedDirection(nn.Module):
    def __init__(self, weight, rank=64):
        super().__init__()
        rank = min(rank, *weight.shape)
        self.a = nn.Parameter(torch.empty(rank, weight.shape[1]))
        self.b = nn.Parameter(torch.zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.magnitude = nn.Parameter(weight.detach().float().norm(dim=1))
        self.register_buffer("base_squared_norm", self.magnitude.detach().square())
        self.register_buffer("importance", torch.zeros(rank))
        self.register_buffer("order", torch.arange(rank))
        self.register_buffer("cap", torch.tensor(rank))
        self.active = rank
        self.register_load_state_dict_post_hook(self._restore_cap)

    @staticmethod
    def _restore_cap(module, incompatible):
        module.active = int(module.cap)

    def forward(self, base, x):
        k = int(torch.randint(1, self.active + 1, ())) if self.training else self.active
        ids = self.order[:k]
        a, b = self.a[ids], self.b[:, ids]
        # Eq. 2, evaluating the same row norm via the small rank Gram matrix.
        # This avoids materializing a dense trainable weight for each task.
        with torch.autocast(x.device.type, enabled=False):
            a32, b32 = a.float(), b.float() * 4
            cross = ((base.weight.float() @ a32.T) * b32).sum(1)
            residual_norm = ((b32 @ (a32 @ a32.T)) * b32).sum(1)
            norm = (self.base_squared_norm + 2 * cross + residual_norm).clamp_min(1e-12).sqrt()
            gain = self.magnitude / norm
        delta = F.linear(F.linear(F.dropout(x, 0.05, self.training), a), b)
        value = F.linear(x, base.weight) + 4 * delta
        value = value * gain.to(value.dtype)
        return value if base.bias is None else value + base.bias.to(value.dtype)

    @torch.no_grad()
    def score(self):
        if self.a.grad is None or self.b.grad is None:
            return
        score = ((self.a * self.a.grad).sum(1).abs() + (self.b * self.b.grad).sum(0).abs()) * 0.5
        self.importance.lerp_(score, 0.05)

    @torch.no_grad()
    def shrink(self):
        ids = self.order[: self.active]
        scores, indices = self.importance[ids].sort(descending=True, stable=True)
        if scores.sum() <= 0:
            return
        k = int(torch.searchsorted(scores.cumsum(0), scores.sum() * 0.95)) + 1
        self.order[: self.active] = ids[indices]
        self.active = min(self.active, max(k + 1, 2))
        self.cap.fill_(self.active)


class RankProjection(nn.Module):
    def __init__(self, base, groups=()):
        super().__init__()
        self.base = base
        self.shared = RankedDirection(base.weight)
        self.private = nn.ModuleDict({g: RankedDirection(base.weight) for g in groups})

    def forward(self, x, task=None):
        adapter = self.shared if task is None else self.private[task]
        return adapter(self.base, x)


def adapt_rank_block(block, stage, groups):
    for name in ("q", "k", "v"):
        setattr(block.attn, name, RankProjection(getattr(block.attn, name)))
    if stage:
        return TaskExitBlock(block, groups, True, lambda base: RankProjection(base, groups))
    block.attn.proj = RankProjection(block.attn.proj)
    block.mlp.fc1 = RankProjection(block.mlp.fc1)
    block.mlp.fc2 = RankProjection(block.mlp.fc2)
    return block


@HOOKS.register_module()
class FAARRankHook(Hook):
    priority = "HIGH"

    def before_train(self, runner):
        self.adapters = [m for m in runner.model.modules() if isinstance(m, RankedDirection)]
        if not self.adapters:
            raise ValueError("No FAAR adapters to rank")
        self.handle = runner.optim_wrapper.optimizer.register_step_pre_hook(self._score)

    def _score(self, optimizer, args, kwargs):
        for m in self.adapters:
            m.score()

    def after_train_epoch(self, runner):
        for m in self.adapters:
            m.shrink()
        caps = torch.tensor([m.active for m in self.adapters])
        runner.logger.info(
            "FAAR active ranks min=%d mean=%.2f max=%d", caps.min(), caps.float().mean(), caps.max()
        )

    def after_train(self, runner):
        self.handle.remove()


class SpectralConsensus(nn.Module):
    """Per-channel radial filter, then low-amplitude and high-edge consensus."""

    def __init__(self, channels):
        super().__init__()
        self.bands = nn.Parameter(torch.randn(16, channels) * 0.02)
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(channels, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.6))
        self.strength = nn.Parameter(torch.zeros(2))
        self.anchor = nn.Parameter(torch.ones(2))

    def forward(self, x, donors):
        h, w = x.shape[-2:]
        # Native unshifted rFFT ordering places DC at (0,0).
        y = 2 * torch.fft.fftfreq(h, device=x.device)[:, None]
        u = 2 * torch.fft.rfftfreq(w, device=x.device)[None, :]
        radius = (y.square() + u.square()).sqrt().clamp(max=1)
        bins = (radius * 16).long().clamp(max=15)
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        filtered = spectrum * (1 + self.bands[bins].permute(2, 0, 1))
        spatial = torch.fft.irfft2(filtered, s=(h, w), norm="ortho")
        own = torch.fft.rfft2(
            x.float() + self.alpha * (spatial * self.scale + self.shift), norm="ortho"
        )
        donor = torch.fft.rfft2(torch.stack(donors).float().mean(0), norm="ortho")
        low, edge = radius <= 0.5, ((radius - 0.8) / 0.2).clamp(min=0)
        gain = self.strength.sigmoid()
        low_delta = own / (own.abs() + 1e-6) * (self.anchor[0] * donor.abs() - own.abs())
        result = (
            own
            + 0.5 * gain[0] * low * low_delta
            + 0.3 * gain[1] * edge * (self.anchor[1] * donor - own)
        )
        return torch.fft.irfft2(result, s=(h, w), norm="ortho").to(x.dtype)


@MODELS.register_module()
class FAARDecoder(DiTASKStateFusion):
    def _build_depth_fusion(self, num_levels, channels):
        super()._build_depth_fusion(num_levels, channels)
        self.filters = nn.ModuleDict(
            {
                g: nn.ModuleList(SpectralConsensus(layer.out_channels) for layer in proj)
                for g, proj in self.adapters.items()
            }
        )

    def states(self, prepared, bounds, groups):
        projected = {}
        for g, seq in prepared.items():
            levels = self.pool_levels(seq, bounds)
            projected[g] = [
                layer(x + seq[1]).masked_fill(~seq[2], 0)
                for layer, x in zip(self.adapters[g], levels)
            ]
        states = {}
        for g in groups:
            filtered = [
                layer(x, [v[i] for other, v in projected.items() if other != g])
                for i, (layer, x) in enumerate(zip(self.filters[g], projected[g]))
            ]
            states[g] = self.fusion[g](torch.cat(filtered, dim=1))
        return states, prepared["coverage"][2]
