"""Single-layer CGC: each task mixes shared experts and its own private expert.

Follows LibMTL's CGC expert connectivity and softmax mixture. Dense spatial
experts replace whole encoders; masked mean state replaces flattened image
input to the gate. This is a spatial adaptation, not full multi-layer PLE.
"""

import torch
from torch import nn

from .branches import StateExpert


class CGCRouter(nn.Module):
    def __init__(self, experts, state_channels=256, channels=64, num_shared=2):
        super().__init__()
        self.experts = tuple(experts)
        if not self.experts or type(num_shared) is not int or num_shared < 1:
            raise ValueError("CGC requires private tasks and positive shared expert count")
        self.shared = nn.ModuleList([
            StateExpert(state_channels, channels) for _ in range(num_shared)])
        self.gates = nn.ModuleDict({
            name: nn.Linear(state_channels, num_shared + 1) for name in self.experts})

    def forward(self, state, features, valid):
        if set(features) != set(self.experts):
            raise ValueError("CGC features disagree with task experts")
        masked = state.masked_fill(~valid, 0)
        pooled = masked.sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)
        shared = [expert(masked, valid) for expert in self.shared]
        outputs = {}
        for task in self.experts:
            # No other task's private expert enters this gate or receives its loss.
            candidates = torch.stack([*shared, features[task]], dim=1)
            weights = self.gates[task](pooled).float().softmax(-1).to(candidates.dtype)
            outputs[task] = torch.einsum("bk,bkchw->bchw", weights, candidates).masked_fill(
                ~valid, 0)
        return outputs
