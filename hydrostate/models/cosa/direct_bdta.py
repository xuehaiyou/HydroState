"""Multi-source deformable G2T attention and a shared per-source T2G update."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .bdta import BidirectionalTaskAdaptation
from .multifm_bdta import MultiFMBDTA, MultiFMBDTADecoder
from .rutal import DeformableCrossAttention


def source_mean(values, valid):
    """Average source-first, channels-last maps at co-registered valid centers."""
    count = valid.sum(0).clamp_min(1)
    return values.masked_fill(~valid[..., None], 0).sum(0) / count[..., None].to(values.dtype)


class DirectSourceAttention(DeformableCrossAttention):
    def __init__(self, channels, heads, points=4, sources=2):
        super().__init__(channels, heads, points=points)
        self.extra_offsets = nn.ModuleList(copy.deepcopy(self.offset) for _ in range(sources - 1))
        self.extra_weights = nn.ModuleList(copy.deepcopy(self.weight) for _ in range(sources - 1))

    @staticmethod
    def point_weights(logits, supported, learn_source_weights):
        # S,B,H,W,heads,P. logsumexp gives the exact source marginal of a
        # joint softmax over (source, point). The control changes only this marginal.
        if learn_source_weights:
            mass = logits.logsumexp(-1).masked_fill(~supported, torch.finfo(logits.dtype).min)
            mass = mass.softmax(0).masked_fill(~supported, 0)
        else:
            mass = supported.float() / supported.sum(0).clamp_min(1)
        return mass[..., None] * logits.softmax(-1)

    def forward(self, q, kv, support):
        valid, ids, learn_source_weights = support
        s, b, h, w, c = kv.shape
        heads, points = self.heads, self.points
        offsets = (self.offset, *self.extra_offsets)
        scores = (self.weight, *self.extra_weights)
        grids = torch.cat([self.sampling_grid(q, offsets[i](q)) for i in ids])
        logits = torch.stack([scores[i](q) for i in ids]).reshape(s, b, h, w, heads, points)
        value = self.value(kv).masked_fill(~valid[..., None], 0)
        value = value.reshape(s, b, h, w, heads, c // heads).permute(0, 1, 4, 5, 2, 3)
        # Sample the validity channel on exactly the same grid as the features.
        support_map = valid[:, :, None, None].expand(-1, -1, heads, 1, -1, -1)
        packed = torch.cat([value.float(), support_map.float()], dim=3).flatten(0, 2)
        sampled = F.grid_sample(packed, grids, align_corners=False)
        sampled = sampled.reshape(s, b, heads, c // heads + 1, h, w, points)
        supported = (sampled[:, :, :, -1] > 0).any(-1).permute(0, 1, 3, 4, 2)
        weights = self.point_weights(logits.float(), supported, learn_source_weights)
        weights = weights.permute(0, 1, 4, 2, 3, 5).unsqueeze(3)
        merged = (sampled[:, :, :, :-1] * weights).sum((0, -1))
        return self.output(merged.permute(0, 3, 4, 1, 2).reshape(b, h, w, c).to(q.dtype))


class DirectBDTAStage(BidirectionalTaskAdaptation):
    g2t_attention_type = DirectSourceAttention

    def forward(self, previous, sources, support):
        available = support[0]
        general = self.general_down(sources)
        valid = available.any(0)
        # Reverse attention reads the common task map, and only valid FM queries
        # contribute to its return. Folding source into batch shares all T2G parameters.
        s, b = general.shape[:2]
        task_mask = valid[:, None].expand(s, -1, -1, -1, -1).flatten(0, 1)
        states = {}
        for task, previous_state in previous.items():
            adapted = general + self.g2t_adapter[task](general)
            task_state = self.g2t(self.task_down(previous_state), adapted, support)
            adapted = task_state + self.t2g_adapter[task](task_state)
            keys = adapted.expand(s, *adapted.shape).flatten(0, 1)
            returned = self.t2g(general.flatten(0, 1), keys, task_mask).unflatten(0, (s, b))
            combined = task_state + source_mean(returned, available)
            states[task] = self.restore(combined).masked_fill(~valid[..., None], 0)
        return states


@MODELS.register_module()
class DirectBDTADecoder(MultiFMBDTADecoder):
    stage_type = DirectBDTAStage
    source_names = ("shared", "anysat")

    def __init__(self, learn_source_weights=True, **kwargs):
        super().__init__(learn_source_weights=False, **kwargs)
        self.learn_source_weights = learn_source_weights

    def states(self, prepared, bounds, groups):
        maps, available = self.pooled_sources(prepared, bounds)
        ids = tuple(self.source_names.index(k) for k in prepared)
        return self.decode_levels(
            source_mean(maps[0], available),
            maps[1:],
            available.any(0)[:, None],
            groups,
            (available, ids, self.learn_source_weights),
        )

    def adapt_level(self, stage, states, feature, valid, source_context):
        updated = stage(states, feature, source_context)
        return updated, source_mean(feature, source_context[0])


@MODELS.register_module()
class DirectMultiFMBDTA(MultiFMBDTA):
    decoder_class = DirectBDTADecoder
