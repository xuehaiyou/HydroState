"""Single-layer CGC: each task mixes shared experts and its own private expert.

Follows LibMTL's CGC expert connectivity and softmax mixture. Dense spatial
experts replace whole encoders; masked mean state replaces flattened image
input to the gate. This is a spatial adaptation, not full multi-layer PLE.
"""

import torch
from torch import nn

from .branches import StateExpert


def split_gate_weights(shared_logits, private_logits):
    """Conditional weights over shared experts and marginal probability of private use."""
    q = shared_logits[..., :-1].float().softmax(-1)
    private = private_logits.float().softmax(-1)[..., -1:]
    shared = q * (1 - private)
    return torch.cat((shared, private.expand_as(shared[..., :1])), dim=-1)


class CGCRouter(nn.Module):
    def __init__(self, experts, state_channels=256, channels=64, num_shared=2,
                 spatial_gates=(), private_only=(), split_gates=None):
        super().__init__()
        self.experts = tuple(experts)
        self.spatial_gates = frozenset(spatial_gates)
        self.private_only = frozenset(private_only)
        self.split_gates = split_gates
        if split_gates is not None and (
            self.spatial_gates or self.private_only or split_gates not in
            ('tile_tile', 'tile_local', 'local_tile', 'local_local')
        ):
            raise ValueError('Expected two gate scopes in shared/private order')
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
        mixed, _ = self.extract(state, features, valid)
        return mixed

    def extract(self, state, features, valid, selectors=None, locality=None):
        if set(features) != set(self.experts):
            raise ValueError("CGC features disagree with task experts")
        masked = state.masked_fill(~valid, 0)
        count = valid.sum((-2, -1)).clamp_min(1)
        pooled_state = masked.sum((-2, -1)) / count if selectors is None else None
        shared = [expert(masked, valid) for expert in self.shared]
        outputs = {}
        for task in self.experts:
            if task in self.private_only:
                outputs[task] = features[task].masked_fill(~valid, 0)
                continue
            # No other task's private expert enters this gate or receives its loss.
            candidates = torch.stack([*shared, features[task]], dim=1)
            selector = masked if selectors is None else selectors[task].masked_fill(~valid, 0)
            if self.split_gates is not None:
                shared_scope, private_scope = self.split_gates.split('_')
                pooled = pooled_state if selectors is None else selector.sum((-2, -1)) / count
                scores = {}
                gate = self.gates[task]
                if 'tile' in (shared_scope, private_scope):
                    scores['tile'] = gate(pooled)[:, None, None, :]
                if 'local' in (shared_scope, private_scope):
                    scores['local'] = gate(selector.movedim(1, -1))
                weights = split_gate_weights(scores[shared_scope], scores[private_scope])
                outputs[task] = ((weights.movedim(-1, 1)[:, :, None].to(candidates.dtype)
                                  * candidates).sum(1)).masked_fill(~valid, 0)
                continue
            if task in self.spatial_gates or locality is not None:
                logits = self.gates[task](selector.movedim(1, -1))
                if locality is not None:
                    # A linear gate commutes with the masked spatial mean.
                    logits = logits.float()
                    mean = logits.masked_fill(~valid.movedim(1, -1), 0).sum((1, 2)) / count
                    logits = torch.lerp(mean[:, None, None, :], logits, locality[task].sigmoid().float())
                weights = logits.float().softmax(-1).to(candidates.dtype)
                outputs[task] = torch.einsum("bhwk,bkchw->bchw", weights, candidates).masked_fill(
                    ~valid, 0)
                continue
            pooled = pooled_state if selectors is None else selector.sum((-2, -1)) / count
            weights = self.gates[task](pooled).float().softmax(-1).to(candidates.dtype)
            outputs[task] = torch.einsum("bk,bkchw->bchw", weights, candidates).masked_fill(
                ~valid, 0)
        return outputs, shared


class RecurrentCGCRouter(CGCRouter):
    """Repeat spatial expert extraction with a shared set of refinement weights."""

    def __init__(self, experts, state_channels=256, channels=64, num_shared=2,
                 spatial_gates=('sm',), depth=3, learn_locality=False):
        if type(depth) is not int or depth < 1:
            raise ValueError("CGC depth must be a positive integer")
        super().__init__(experts, state_channels, channels, num_shared, spatial_gates)
        if learn_locality and self.spatial_gates:
            raise ValueError("Choose learned locality or fixed spatial gates")
        self.locality = (nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in experts})
                         if learn_locality else None)
        self.depth = depth
        if depth > 1:
            self.refinement = CGCRouter(experts, channels, channels, num_shared, spatial_gates)
            self.private = nn.ModuleDict({name: StateExpert(channels, channels) for name in experts})
            self.step = nn.Parameter(torch.zeros(()))

    def forward(self, state, features, valid):
        tasks, shared = self.extract(state, features, valid, locality=self.locality)
        if self.depth == 1:
            return tasks
        common = torch.stack(shared).mean(0)
        for _ in range(self.depth - 1):
            candidates = {name: expert(tasks[name], valid) for name, expert in self.private.items()}
            proposal, shared = self.refinement.extract(common, candidates, valid, selectors=tasks,
                                                        locality=self.locality)
            tasks = {name: value + self.step * (proposal[name] - value)
                     for name, value in tasks.items()}
            common = common + self.step * (torch.stack(shared).mean(0) - common)
        return tasks
