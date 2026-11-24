"""RUTaL FFN residual upcycling and the four-stage task ladder (Yao et al., Eqs. 5–7)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..shared_private_lora import LowRankUpdate
from .state import CoSAStateFusion


class ResidualExpertFFN(nn.Module):
    def __init__(self, mlp, rank=8, experts=4):
        super().__init__()
        self.base = mlp
        c, hidden = mlp.fc1.in_features, mlp.fc1.out_features
        self.expand = LowRankUpdate(c, hidden, rank, scale=1.0)
        self.contract = nn.ModuleList(
            LowRankUpdate(hidden, c, rank, scale=1.0) for _ in range(experts)
        )
        self.router = nn.Linear(c, experts, bias=False)

    def forward(self, x):
        weights = self.router(x).float().softmax(-1).to(x.dtype)
        hidden = self.base.drop1(self.base.act(self.base.fc1(x) + self.expand(x)))
        value = self.base.fc2(hidden)
        for i, expert in enumerate(self.contract):
            value = value + weights[..., i : i + 1] * expert(hidden)
        return self.base.drop2(value)


def projector(in_channels, out_channels):
    hidden = max(out_channels // 4, 1)
    return nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, out_channels))


class DeformableCrossAttention(nn.Module):
    """Single-level deformable attention on pixel centers with learned offsets per head."""

    def __init__(self, channels, heads=6, points=4):
        super().__init__()
        if channels % heads:
            raise ValueError("Ladder width must divide across attention heads")
        self.heads, self.points = heads, points
        self.offset = nn.Linear(channels, heads * points * 2)
        self.weight = nn.Linear(channels, heads * points)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        nn.init.zeros_(self.offset.weight)
        angle = torch.arange(heads) * (2 * math.pi / heads)
        ray = torch.stack((angle.cos(), angle.sin()), dim=-1)
        ray = ray / ray.abs().amax(-1, keepdim=True)
        with torch.no_grad():
            self.offset.bias.copy_(
                (ray[:, None] * torch.arange(1, points + 1)[None, :, None]).flatten()
            )
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, q, kv, valid):
        b, h, w, c = q.shape
        heads, points = self.heads, self.points
        y, x = torch.meshgrid(
            torch.arange(h, device=q.device), torch.arange(w, device=q.device), indexing="ij"
        )
        centers = torch.stack((x, y), -1).float()[None, :, :, None, None] + 0.5
        offsets = self.offset(q).reshape(b, h, w, heads, points, 2).float()
        grid = (centers + offsets) * q.new_tensor([2 / w, 2 / h]).float() - 1
        grid = grid.permute(0, 3, 1, 2, 4, 5).reshape(b * heads, h, w * points, 2)
        values = self.value(kv).masked_fill(~valid.permute(0, 2, 3, 1), 0)
        values = values.reshape(b, h, w, heads, c // heads).permute(0, 3, 4, 1, 2)
        sampled = F.grid_sample(
            values.reshape(b * heads, c // heads, h, w).float(), grid, align_corners=False
        ).reshape(b, heads, c // heads, h, w, points)
        weights = self.weight(q).reshape(b, h, w, heads, points).float().softmax(-1)
        merged = (sampled * weights.permute(0, 3, 1, 2, 4)[:, :, None]).sum(-1)
        return self.output(merged.permute(0, 3, 4, 1, 2).reshape(b, h, w, c).to(q.dtype))


class LadderStage(nn.Module):
    def __init__(self, channels, groups, heads):
        super().__init__()
        d = channels // 4
        self.project = nn.ModuleDict({g: projector(channels, d) for g in groups})
        self.modulate = nn.ModuleDict({g: projector(channels, 2 * d) for g in groups})
        self.reduce = nn.Linear(channels, d)
        self.query_norm, self.key_norm, self.ffn_norm = (nn.LayerNorm(d) for _ in range(3))
        self.attention = DeformableCrossAttention(d, heads)
        self.up, self.down = nn.Linear(d, d), nn.Linear(d, d)
        self.restore = nn.Linear(d, channels)

    def forward(self, group, previous, feature, valid):
        q = self.reduce(previous)
        z = self.project[group](feature)
        gamma, beta = self.modulate[group](feature).chunk(2, -1)
        h = q + self.attention(self.query_norm(q), self.key_norm(z), valid)
        return self.restore(self.down(gamma * F.gelu(self.up(self.ffn_norm(h))) + beta))


@MODELS.register_module()
class RUTaLDecoder(CoSAStateFusion):
    def __init__(self, ladder_heads=6, **kwargs):
        self.ladder_heads = ladder_heads
        super().__init__(**kwargs)

    def _build_depth_fusion(self, num_levels, channels):
        if num_levels != 5:
            raise ValueError("RUTaL takes the initial patch map and four stage features")
        groups = ("coverage", "sm", "et")
        self.stages = nn.ModuleList(
            LadderStage(channels, groups, self.ladder_heads) for _ in range(4)
        )
        self.level_readouts = nn.ModuleDict(
            {
                group: nn.ModuleList(nn.Linear(channels, channels) for _ in self.stages)
                for group in groups
            }
        )
        self.head_channels = channels // 2
        self.readouts = nn.ModuleDict(
            {
                g: nn.Sequential(
                    nn.Conv2d(4 * channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.ConvTranspose2d(channels, self.head_channels, 2, stride=2),
                    nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),
                    nn.ReLU(),
                )
                for g in groups
            }
        )

    def states(self, prepared, bounds, groups):
        seq = prepared["shared"]
        levels = self.pool_levels(seq, bounds) + seq[1][None]
        valid = seq[2]
        patch, *features = levels.permute(0, 1, 3, 4, 2).unbind()
        states = {}
        for group in groups:
            current, outputs = patch, []
            for stage, feature, readout in zip(self.stages, features, self.level_readouts[group]):
                current = stage(group, current, feature, valid).masked_fill(
                    ~valid.permute(0, 2, 3, 1), 0
                )
                outputs.append(
                    readout(current).masked_fill(~valid.permute(0, 2, 3, 1), 0).permute(0, 3, 1, 2)
                )
            states[group] = self.readouts[group](torch.cat(outputs, dim=1))
        mask = F.interpolate(valid.float(), scale_factor=2, mode="nearest").bool()
        return states, mask
