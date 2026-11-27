"""Task-conditioned source selection on the general-feature side of BDTA."""

import copy
from functools import partial

import torch
from torch import nn

from hydrostate.registry import MODELS

from .bdta import BDTAMoTEDecoder
from .multifm import MultiFMStageModel, source_weights


@MODELS.register_module()
class MultiFMBDTADecoder(BDTAMoTEDecoder):
    def __init__(self, learn_source_weights=True, secondary_widths=None, **kwargs):
        super().__init__(**kwargs)
        self.anysat_projections = (
            copy.deepcopy(self.projections) if secondary_widths is None else nn.ModuleList(
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.out_channels))
                for width in secondary_widths
            )
        )
        self.source_scale = nn.Parameter(torch.zeros(())) if learn_source_weights else None

    def source_context(self, states, features, available):
        # The same source rule serves every task, depth and pixel. Both BDTA directions
        # and the recipient's MoTE router see the same general feature.
        contexts = {}
        for task, query in states.items():
            weights = source_weights(query, features, available, self.source_scale)
            contexts[task] = (features * weights[..., None].to(features.dtype)).sum(0)
        return contexts

    def pooled_sources(self, prepared, bounds):
        # Each FM first pools its own observations against the recipient's time interval.
        maps = torch.stack([
            (self.pool_levels(seq, bounds) + seq[1][None]).permute(0, 1, 3, 4, 2)
            for seq in prepared.values()
        ], dim=1)  # level, source, B, H, W, C
        available = torch.stack([seq[2][:, 0] for seq in prepared.values()])
        maps = maps.masked_fill(~available[None, ..., None], 0)
        return maps, available

    def states(self, prepared, bounds, groups):
        maps, available = self.pooled_sources(prepared, bounds)
        patch = maps[0].sum(0) / available.sum(0).clamp_min(1)[..., None].to(maps.dtype)
        valid = available.any(0)[:, None]
        return self.decode_levels(
            patch, maps[1:], valid, groups, partial(self.source_context, available=available)
        )


@MODELS.register_module()
class MultiFMBDTA(MultiFMStageModel):
    decoder_class = MultiFMBDTADecoder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.anysat.frozen:
            raise ValueError("Multi-FM BDTA requires both pretrained encoders to be frozen")
