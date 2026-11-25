"""Four RUTaL stages with a masked choice of foundation model at each sampled location."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hydrostate.registry import MODELS

from .multifm import MultiFMStageModel, source_weights
from .rutal import LadderStage, RUTaLDecoder


class MultiSourceStage(LadderStage):
    def forward(self, group, previous, features, masks, scale):
        # features: sources,B,H,W,C; masks: sources,B,1,H,W
        q = self.reduce(previous)
        query = self.query_norm(q)
        keys = self.key_norm(self.project[group](features))
        center_valid = masks.squeeze(2)
        center_weights = source_weights(query, keys, center_valid, scale)
        center = (features * center_weights[..., None].to(features.dtype)).sum(0)
        context = self.read_locations(query, keys, masks, scale)
        return self.update_state(group, q, context, center)

    def read_locations(self, query, keys, masks, scale):
        att = self.attention
        sources, batch, height, width, channels = keys.shape
        heads, points = att.heads, att.points
        dim = channels // heads
        grid = att.sampling_grid(query)

        # Pack one source and head per grid_sample batch. Interpolate support at the
        # same fractional coordinates as features, including the image boundary.
        def per_head(x):
            return x.reshape(sources, batch, height, width, heads, dim).permute(0, 1, 4, 5, 2, 3)

        support = masks.unsqueeze(2).expand(-1, -1, heads, -1, -1, -1)
        fields = [per_head(att.value(keys))]
        if scale is not None:
            fields.append(per_head(keys))
        packed = torch.cat([*fields, support.to(keys.dtype)], dim=3).masked_fill(~support, 0)
        sampled = (
            F.grid_sample(
                packed.flatten(0, 2).float(), grid.repeat(sources, 1, 1, 1), align_corners=False
            )
            .reshape(sources, batch, heads, -1, height, width, points)
            .movedim(3, -1)
        )
        values = sampled[..., :dim]
        valid = sampled[..., -1] > 0
        q_heads = query.reshape(batch, height, width, heads, dim).permute(0, 3, 1, 2, 4)
        weights = source_weights(q_heads.unsqueeze(-2), sampled[..., dim:-1], valid, scale)
        mixed = (values * weights[..., None]).sum(0)
        spatial = att.weight(query).reshape(batch, height, width, heads, points).float().softmax(-1)
        result = (mixed * spatial.permute(0, 3, 1, 2, 4)[..., None]).sum(-2)
        return att.output(result.permute(0, 2, 3, 1, 4).reshape_as(query).to(query.dtype))


@MODELS.register_module()
class MultiFMRUTaLDecoder(RUTaLDecoder):
    def __init__(self, learn_source_weights=True, **kwargs):
        super().__init__(**kwargs)
        # Each pretrained representation gets its own trainable channel alignment.
        self.anysat_projections = copy.deepcopy(self.projections)
        self.source_scale = nn.Parameter(torch.zeros(())) if learn_source_weights else None

    stage_type = MultiSourceStage

    def states(self, prepared, bounds, groups):
        levels = torch.stack([self.pool_levels(s, bounds) + s[1][None] for s in prepared.values()])
        masks = torch.stack([s[2] for s in prepared.values()])
        valid = masks.any(0)
        maps = levels.permute(1, 0, 2, 4, 5, 3)
        count = masks.sum(0).clamp_min(1).permute(0, 2, 3, 1)
        patch = (maps[0] * masks.permute(0, 1, 3, 4, 2)).sum(0) / count
        states = {}
        for group in groups:
            current, outputs = patch, []
            for stage, feature, readout in zip(self.stages, maps[1:], self.level_readouts[group]):
                args = (group, current, feature, masks, self.source_scale)
                current = (
                    checkpoint(stage, *args, use_reentrant=False) if self.training else stage(*args)
                )
                current = current.masked_fill(~valid.permute(0, 2, 3, 1), 0)
                outputs.append(
                    readout(current).masked_fill(~valid.permute(0, 2, 3, 1), 0).permute(0, 3, 1, 2)
                )
            states[group] = self.readouts[group](torch.cat(outputs, dim=1))
        return states, F.interpolate(valid.float(), scale_factor=2, mode="nearest").bool()


@MODELS.register_module()
class MultiFMRUTaL(MultiFMStageModel):
    decoder_class = MultiFMRUTaLDecoder
