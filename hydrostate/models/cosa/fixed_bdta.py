"""The existing BDTA network with equal group weights and fixed source weights."""

import torch

from hydrostate.models.cosa.foundation_v2 import FoundationStageModelV2
from hydrostate.registry import MODELS


@MODELS.register_module()
class FixedSupervisionBDTA(FoundationStageModelV2):
    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        terms = super()._loss_from_outputs(outputs, data_samples, auxiliary, auxiliary_weights)
        for group, tasks in self.expert_groups.items():
            present = torch.stack([terms[f"valid_count_{t}"] > 0 for t in tasks])
            errors = torch.stack([terms.pop(f"loss_{t}") for t in tasks])
            terms[f"loss_product_{group}"] = errors.sum() / (9 * present.sum().clamp_min(1))
        return terms
