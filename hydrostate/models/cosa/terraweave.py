"""Stage-specific reciprocal attention for time-aligned, frozen foundation-model features."""

import copy

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .state import CoSAStateFusion


def partition(x, side):
    """B,S,H,W,C -> windows,S,positions,C. Pad only the bottom and right edges."""
    height, width = x.shape[2:4]
    x = F.pad(x, (0, 0, 0, -width % side, 0, -height % side))
    return rearrange(x, "b s (h i) (w j) c -> (b h w) s (i j) c", i=side, j=side)


def unpartition(x, height, width, side):
    rows, cols = (height + side - 1) // side, (width + side - 1) // side
    x = rearrange(x, "(b h w) s (i j) c -> b s c (h i) (w j)", h=rows, w=cols, i=side, j=side)
    return x[..., :height, :width]


class UnifiedReciprocalAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.out = nn.Linear(channels, channels, bias=False)
        self.gamma_return = nn.Parameter(torch.tensor(0.1))
        self.gamma_variable = nn.Parameter(torch.tensor(0.1))
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def heads_of(self, x):
        return rearrange(x, "b n (h d) -> b h n d", h=self.heads)

    def weights(self, queries, keys, valid):
        score = (queries @ keys.transpose(-1, -2)).float() * self.scale
        valid = valid[:, None, None, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        return score.softmax(-1).masked_fill(~valid, 0).to(queries.dtype)

    def project_output(self, x):
        return self.out(rearrange(x, "b h n d -> b n (h d)"))

    def forward(self, state, evidence, variable_valid, evidence_valid):
        z, e = self.norm(state), self.norm(evidence)
        a = self.weights(self.heads_of(self.q(z)), self.heads_of(self.k(e)), evidence_valid)
        u = state + self.project_output(a @ self.heads_of(self.v(e)))
        normalized = self.norm(u)
        # Row-wise normalization allows the reverse and variable queries to share keys and values.
        queries = self.heads_of(self.q(torch.cat((e, normalized), dim=1)))
        b = self.weights(queries, self.heads_of(self.k(normalized)), variable_valid)
        messages = b @ self.heads_of(self.v(normalized))
        reverse, direct = messages.split((e.shape[1], u.shape[1]), dim=2)
        u = u + self.project_output(
            self.gamma_return * (a @ reverse) + self.gamma_variable * direct
        )
        return (u + self.ffn(u)).masked_fill(~variable_valid[..., None], 0)


@MODELS.register_module()
class TerraWeaveDecoder(CoSAStateFusion):
    groups = ("coverage", "sm", "et")

    def __init__(self, channels=128, num_levels=4, num_heads=4, window_size=4, sources=("shared", "anysat"), **kwargs):
        if num_levels != 4 or window_size <= 0:
            raise ValueError("TerraWeave requires four stage exits and a positive window size")
        self.sources = tuple(sources)
        self.window_size = window_size
        self.attention_heads = num_heads
        super().__init__(channels=channels, num_levels=num_levels, num_heads=num_heads, **kwargs)

    def _build_depth_fusion(self, num_levels, channels):
        self.head_channels = channels
        if "anysat" in self.sources:
            self.anysat_projections = copy.deepcopy(self.projections)
        self.source_id = nn.Embedding(len(self.sources), channels)
        self.variable_id = nn.Embedding(len(self.groups), channels)
        self.position = nn.Embedding(self.window_size**2, channels)
        for embedding in (self.source_id, self.variable_id, self.position):
            nn.init.normal_(embedding.weight, std=0.02)
        self.stages = nn.ModuleList(
            UnifiedReciprocalAttention(channels, self.attention_heads) for _ in range(num_levels)
        )
        self.readout = nn.Linear(num_levels * channels, channels)

    def states(self, prepared, bounds, groups):
        source_ids = [self.sources.index(name) for name in prepared]
        # Pool each source/depth on the recipient's interval before constructing spatial tokens.
        maps = torch.stack(
            [self.pool_levels(seq, bounds) + seq[1][None] for seq in prepared.values()], dim=2
        )  # depth,B,source,C,H,W
        side = self.window_size
        height, width = maps.shape[-2:]
        support = torch.stack(
            [
                torch.stack([valid.any(1) for _, valid, _ in levels])
                for levels, _, _ in prepared.values()
            ],
            dim=2,
        )  # depth,B,source,H,W
        position = self.position.weight
        seed = self.variable_id.weight[:, None, :] + position
        source_tags = self.source_id.weight[source_ids, None, :] + position
        readouts = []
        state = seed.flatten(0, 1)[None]
        requested = [self.groups.index(name) for name in groups]
        for stage, features, valid in zip(self.stages, maps, support, strict=True):
            evidence = partition(features.permute(0, 1, 3, 4, 2), side)
            evidence = (evidence + source_tags).flatten(1, 2)
            evidence_valid = partition(valid[..., None], side).squeeze(-1)
            variable_valid = evidence_valid.any(1)[:, None].expand(-1, len(self.groups), -1)
            state = stage(
                state.expand(evidence.shape[0], -1, -1),
                evidence,
                variable_valid.flatten(1, 2),
                evidence_valid.flatten(1, 2),
            )
            readouts.append(state.unflatten(1, (len(self.groups), side**2))[:, requested])
        fused = self.readout(torch.cat(readouts, dim=-1))
        valid = support.any(0).any(1)[:, None]
        output = unpartition(fused, height, width, side).masked_fill(~valid[:, None], 0)
        return dict(zip(groups, output.unbind(1), strict=True)), valid
