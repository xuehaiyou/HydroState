"""Frozen-feature adaptation with BDTA and MoTE (Yao et al., AAAI 2026, Eqs. 3–11)."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .rutal import DeformableCrossAttention
from .state import CoSAStateFusion


def mlp(channels, hidden, out):
    return nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, out))


class CrossUpdate(nn.Module):
    """Equations 5 and 7 use a residual for both attention and the subsequent FFN."""

    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.attention = DeformableCrossAttention(width, heads=heads, points=4)
        self.ffn = nn.Sequential(nn.LayerNorm(width), mlp(width, width, width))

    def forward(self, query, source, valid):
        hidden = query + self.attention(self.query_norm(query), self.key_norm(source), valid)
        return hidden + self.ffn(hidden)


class BidirectionalTaskAdaptation(nn.Module):
    def __init__(self, channels, groups, heads):
        super().__init__()
        width = channels // 4
        bottleneck = max(1, width // 4)
        self.general_down = nn.Linear(channels, width)
        self.task_down = nn.Linear(channels, width)
        self.restore = nn.Linear(width, channels)
        self.g2t_adapter = nn.ModuleDict({g: mlp(width, bottleneck, width) for g in groups})
        self.t2g_adapter = nn.ModuleDict({g: mlp(width, bottleneck, width) for g in groups})
        self.g2t = CrossUpdate(width, heads)
        self.t2g = CrossUpdate(width, heads)

    def forward(self, previous, feature, valid):
        if isinstance(feature, dict):
            contexts = {g: self.general_down(value) for g, value in feature.items()}
        else:
            contexts = dict.fromkeys(previous, self.general_down(feature))
        missing = ~valid.permute(0, 2, 3, 1)
        states = {}
        for task, state in previous.items():
            general = contexts[task]
            adapted = general + self.g2t_adapter[task](general)
            task_state = self.g2t(self.task_down(state), adapted, valid)
            adapted = task_state + self.t2g_adapter[task](task_state)
            general_state = self.t2g(general, adapted, valid)
            states[task] = self.restore(task_state + general_state).masked_fill(missing, 0)
        return states


class TaskRouter(nn.Module):
    """Per-stage and per-recipient scene weights; noise is used only for training."""

    def __init__(self, channels, tasks):
        super().__init__()
        hidden = max(1, channels // 4)
        self.scores = mlp(channels, hidden, tasks)
        self.noise = mlp(channels, hidden, tasks)
        self.gamma = nn.Parameter(torch.zeros(channels))

    def forward(self, feature, valid):
        mask = valid.permute(0, 2, 3, 1)
        count = mask.sum((1, 2)).clamp_min(1)

        def pool(value):
            return value.float().masked_fill(~mask, 0).sum((1, 2)) / count

        logits = pool(self.scores(feature))
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(pool(self.noise(feature)))
        return logits.softmax(-1)


class MixtureOfTaskExperts(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = tuple(groups)
        self.project = nn.ModuleDict({g: nn.Linear(channels, channels) for g in groups})
        self.routers = nn.ModuleDict({g: TaskRouter(channels, len(groups)) for g in groups})

    def forward(self, states, feature, valid, recipients):
        experts = torch.stack([self.project[g](states[g]) for g in self.groups], dim=1)
        mixed = {}
        for g in recipients:
            router = self.routers[g]
            context = feature[g] if isinstance(feature, dict) else feature
            weights = router(context, valid).to(experts.dtype)
            transfer = (experts * weights[:, :, None, None, None]).sum(1)
            mixed[g] = experts[:, self.groups.index(g)] + router.gamma.to(transfer.dtype) * transfer
        return mixed


@MODELS.register_module()
class BDTAMoTEDecoder(CoSAStateFusion):
    def __init__(self, bypass_heads=6, **kwargs):
        self.bypass_heads = bypass_heads
        super().__init__(**kwargs)

    def _build_depth_fusion(self, num_levels, channels):
        if num_levels != 5 or channels % 4:
            raise ValueError("BDTA needs initial patches, four stages and a quarter-width bypass")
        self.groups = ("coverage", "sm", "et")
        self.stages = nn.ModuleList(
            BidirectionalTaskAdaptation(channels, self.groups, self.bypass_heads) for _ in range(4)
        )
        self.mixtures = nn.ModuleList(MixtureOfTaskExperts(channels, self.groups) for _ in range(4))
        # Equation 9 projects to the first stage's width and spatial grid.
        # All four OlmoEarth exits already share both dimensions.
        self.head_channels = channels
        self.fuse = nn.ModuleDict({g: nn.Linear(4 * channels, channels) for g in self.groups})

    def states(self, prepared, bounds, groups):
        seq = prepared["shared"]
        valid = seq[2]
        levels = self.pool_levels(seq, bounds) + seq[1][None]
        patch, *features = levels.permute(0, 1, 3, 4, 2).unbind()
        return self.decode_levels(patch, features, valid, groups)

    def decode_levels(self, patch, features, valid, groups, source_context=None):
        # All source experts use the recipient interval, even for a single requested task.
        states = dict.fromkeys(self.groups, patch)
        readouts = {g: [] for g in groups}
        for stage, mixture, feature in zip(self.stages, self.mixtures, features, strict=True):
            if source_context is not None:
                feature = source_context(states, feature)
            states = stage(states, feature, valid)
            for g, value in mixture(states, feature, valid, groups).items():
                readouts[g].append(value)
        return {
            g: self.fuse[g](torch.cat(values, -1)).permute(0, 3, 1, 2).masked_fill(~valid, 0)
            for g, values in readouts.items()
        }, valid
