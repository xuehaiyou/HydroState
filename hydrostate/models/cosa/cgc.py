"""Single-layer CGC: each task mixes shared experts and its own private expert.

Follows LibMTL's CGC expert connectivity and softmax mixture. Dense spatial
experts replace whole encoders; masked mean state replaces flattened image
input to the gate. This is a spatial adaptation, not full multi-layer PLE.
"""

import torch
from torch import nn

from .branches import StateExpert


class CGCRouter(nn.Module):
    def __init__(self, experts, state_channels=256, channels=64, num_shared=2,
                 spatial_gates=(), private_only=()):
        super().__init__()
        self.experts = tuple(experts)
        self.spatial_gates = frozenset(spatial_gates)
        self.private_only = frozenset(private_only)
        if not (self.spatial_gates | self.private_only).issubset(self.experts):
            raise ValueError("Routing policies require a declared expert group")
        if self.private_only & self.spatial_gates:
            raise ValueError("A private-only task has no spatial gate")
        if not self.experts or type(num_shared) is not int or num_shared < 1:
            raise ValueError("CGC requires private tasks and positive shared expert count")
        self.shared = nn.ModuleList([
            StateExpert(state_channels, channels) for _ in range(num_shared)])
        self.gates = nn.ModuleDict({
            name: nn.Linear(state_channels, num_shared + 1) for name in self.experts})
        # Consume identical initialization draws for the paired training control.
        for name in self.private_only:
            del self.gates[name]

    def forward(self, state, features, valid):
        if set(features) != set(self.experts):
            raise ValueError("CGC features disagree with task experts")
        masked = state.masked_fill(~valid, 0)
        pooled = masked.sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)
        shared = [expert(masked, valid) for expert in self.shared]
        outputs = {}
        for task in self.experts:
            if task in self.private_only:
                outputs[task] = features[task].masked_fill(~valid, 0)
                continue
            # No other task's private expert enters this gate or receives its loss.
            candidates = torch.stack([*shared, features[task]], dim=1)
            if task in self.spatial_gates:
                logits = self.gates[task](masked.movedim(1, -1))
                weights = logits.float().softmax(-1).to(candidates.dtype)
                outputs[task] = torch.einsum("bhwk,bkchw->bchw", weights, candidates).masked_fill(
                    ~valid, 0)
                continue
            weights = self.gates[task](pooled).float().softmax(-1).to(candidates.dtype)
            outputs[task] = torch.einsum("bk,bkchw->bchw", weights, candidates).masked_fill(
                ~valid, 0)
        return outputs
