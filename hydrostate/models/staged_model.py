"""Staged task specialization, auxiliary supervision and one gated interaction."""

from __future__ import annotations

import copy
import math

from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .hydrostate_model import HydroStateModel
from .task_branches import StagedTaskDecoder


@MODELS.register_module()
class StagedHydroStateModel(HydroStateModel):
    """Expose the same final-output API as HydroStateModel, including predict."""

    def __init__(self, interaction=None, auxiliary_weight=0.2, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.decoder, StagedTaskDecoder):
            raise ValueError("StagedHydroStateModel requires StagedTaskDecoder")
        interaction = dict(interaction or {})
        interaction.setdefault("type", "TaskGatedInteraction")
        interaction.setdefault("tasks", self.task_specs)
        interaction.setdefault("channels", self.decoder.out_channels)
        if interaction["channels"] != self.decoder.out_channels:
            raise ValueError("interaction channels must match decoder output")
        self.interaction = MODELS.build(interaction)
        weights = (
            dict(auxiliary_weight) if isinstance(auxiliary_weight, dict)
            else dict.fromkeys(self.tasks, auxiliary_weight)
        )
        if set(weights) != set(self.tasks):
            raise ValueError(f"auxiliary_weight must specify exactly {self.tasks}")
        self.auxiliary_weights = {task: float(value) for task, value in weights.items()}
        if any(not math.isfinite(v) or v < 0 for v in self.auxiliary_weights.values()):
            raise ValueError("auxiliary weights must be finite and nonnegative")
        # Disabled auxiliary heads have neither parameters nor optimizer state.
        self.auxiliary_heads = nn.ModuleDict({
            task: MODELS.build(copy.deepcopy(kwargs["heads"][task]))
            for task in self.tasks if self.auxiliary_weights[task] > 0
        })

    def _forward_with_auxiliary(self, inputs, auxiliary=False):
        features = self.decoder(self._encode_inputs(inputs))
        pooled = {}
        for task, value in features.items():
            if any(size % self.output_scale for size in value.shape[-2:]):
                raise ValueError("Decoded grid must be divisible by output_scale")
            pooled[task] = F.avg_pool2d(value, self.output_scale)
        aux_outputs = (
            {task: head(pooled[task]) for task, head in self.auxiliary_heads.items()}
            if auxiliary else {}
        )
        interacted = self.interaction(pooled)
        outputs = {task: self.heads[task](interacted[task]) for task in self.tasks}
        return self._constrain_outputs(outputs), self._constrain_outputs(aux_outputs)

    def _forward(self, inputs):
        return self._forward_with_auxiliary(inputs)[0]

    def loss(self, inputs, data_samples):
        outputs, auxiliary = self._forward_with_auxiliary(inputs, auxiliary=True)
        return self._loss_from_outputs(
            outputs, data_samples, auxiliary=auxiliary, auxiliary_weights=self.auxiliary_weights
        )
