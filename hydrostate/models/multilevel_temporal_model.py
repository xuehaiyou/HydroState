"""Multilevel temporal queries, task adapters and interactions at the queried time."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .multilevel_model import MultiLevelFusionDecoder, OutputTaskAdapter
from .temporal_model import TemporalHydroStateModel, TemporalOlmoEarthEncoder


@MODELS.register_module()
class MultiLevelTemporalOlmoEarthEncoder(TemporalOlmoEarthEncoder):
    """Extract three depths in one encoder pass without pooling acquisition times."""

    def __init__(self, out_indices=(3, 7, 11), **kwargs):
        super().__init__(**kwargs)
        self.out_indices = tuple(out_indices)
        if (len(self.out_indices) != 3
                or tuple(sorted(set(self.out_indices))) != self.out_indices
                or self.out_indices[0] < 0
                or self.out_indices[-1] != len(self.backbone.model.blocks) - 1):
            raise ValueError("Select three increasing depths ending at the final encoder block")

    def forward(self, inputs):
        return self.forward_levels(inputs, self.out_indices)


@MODELS.register_module()
class MultiLevelTemporalHydroStateModel(TemporalHydroStateModel):
    """Query each depth before fusion; adapt and exchange features at that same time.

    Each task's output depends on its own query bounds. Source task adapters in
    an interaction are evaluated at the recipient's query, never at another
    output's independently requested date. Exhaustive fractions share a query.
    """

    def __init__(self, adapter=None, interaction=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.decoder, MultiLevelFusionDecoder):
            raise ValueError("Multilevel temporal retrieval requires MultiLevelFusionDecoder")
        channels = self.decoder.out_channels
        self.adapters = nn.ModuleDict({
            task: OutputTaskAdapter(channels=channels, **dict(adapter or {})) for task in self.tasks
        })
        cfg = dict(interaction or {})
        cfg.setdefault("type", "TaskGatedInteraction")
        cfg.setdefault("tasks", self.task_specs)
        cfg.setdefault("channels", channels)
        cfg.setdefault("learnable_scale", True)
        if cfg["channels"] != channels:
            raise ValueError("Interaction channels must match decoder output")
        self.interaction = MODELS.build(cfg)

    def _forward(self, inputs):
        if "target_time" not in inputs or set(inputs["target_time"]) != set(self.tasks):
            raise ValueError("Temporal model requires explicit target_time bounds for every task")
        levels = self.encoder(inputs)
        if not isinstance(levels, (tuple, list)) or len(levels) != 3:
            raise ValueError("Multilevel temporal encoder must return three observation sequences")
        keys = [self.query_time.keys(level["features"].permute(0, 1, 3, 4, 2))
                for level in levels]
        outputs = {}
        for group in self.query_groups:
            bounds = inputs["target_time"][group[0]]
            # Keep the same joint-fraction time invariant as the single-depth model.
            for name in group[1:]:
                torch._assert_async((inputs["target_time"][name] == bounds).all(),
                                    "Joint cover fractions must share a target time")
            features = tuple(self.query_time(level, bounds, group[0], key)
                             for level, key in zip(levels, keys))
            if self.input_fusion is not None:
                features = tuple(self.input_fusion(value, inputs) for value in features)
            decoded = self.decoder(features)
            if any(size % self.output_scale for size in decoded.shape[-2:]):
                raise ValueError("Decoded grid must be divisible by output_scale")
            shared = F.avg_pool2d(decoded, self.output_scale)
            adapted = {task: self.adapters[task](shared) for task in self.tasks}
            interacted = self.interaction(adapted, targets=group)
            outputs.update({task: self.heads[task](interacted[task]) for task in group})
        return self._constrain_outputs(outputs)
