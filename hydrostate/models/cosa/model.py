"""CoSA retrieval with the existing observation-scale losses and prediction API."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..hydrostate_model import HydroStateModel
from ..retrieval_model import RetrievalModelMixin
from .branches import SpatialReadout, StateExpert
from .encoder import CoSASharedLoRAEncoder
from .routing import StateRouter
from .state import CoSAStateFusion


@MODELS.register_module()
class CoSAHydroStateModel(RetrievalModelMixin, HydroStateModel):
    def __init__(self, expert_groups, branches=None, router=None, query_groups=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.encoder, CoSASharedLoRAEncoder):
            raise TypeError("CoSA requires CoSASharedLoRAEncoder")
        if not isinstance(self.decoder, CoSAStateFusion) or self.input_fusion is not None:
            raise ValueError("CoSA uses CoSAStateFusion with its own observation quality encoding")
        if self.decoder.in_channels != self.encoder.out_channels:
            raise ValueError("CoSA state channels disagree with the encoder")
        self.expert_groups = {name: tuple(tasks) for name, tasks in expert_groups.items()}
        assigned = [task for tasks in self.expert_groups.values() for task in tasks]
        if (not self.expert_groups or any(not tasks for tasks in self.expert_groups.values())
                or len(set(assigned)) != len(assigned) or set(assigned) != set(self.tasks)):
            raise ValueError("Expert groups must partition all output tasks exactly once")
        self.query_groups = tuple(tuple(group) for group in (
            [self.tasks] if query_groups is None else query_groups))
        queried = [task for group in self.query_groups for task in group]
        if (any(not group for group in self.query_groups) or len(set(queried)) != len(queried)
                or set(queried) != set(self.tasks)):
            raise ValueError("Query groups must partition output tasks exactly once")
        for group in self.fraction_groups:
            if not any(set(group).issubset(query) for query in self.query_groups):
                raise ValueError("Joint fractions must belong to the same query group")
        options = dict(branches or {})
        channels = options.pop("channels", 64)
        if options or type(channels) is not int or channels < 1:
            raise ValueError("Branches require a positive channels setting")
        self.experts = nn.ModuleDict({
            name: StateExpert(self.decoder.out_channels, channels) for name in self.expert_groups})
        self.readouts = nn.ModuleDict({
            name: SpatialReadout(channels) for name in self.expert_groups})
        self.router = StateRouter(
            self.expert_groups, state_channels=self.decoder.out_channels,
            channels=channels, **dict(router or {}))

    def _forward(self, inputs):
        if set(inputs.get("target_time", {})) != set(self.tasks):
            raise ValueError("CoSA requires explicit target intervals for every output")
        levels = self.encoder(inputs)
        prepared = self.decoder.prepare(levels, inputs)
        size = inputs[self.encoder.input_keys[0]].shape[-2:]
        if any(length % self.output_scale for length in size):
            raise ValueError("Input dimensions must divide by output_scale")
        outputs = {}
        for group in self.query_groups:
            bounds = inputs["target_time"][group[0]]
            for task in group[1:]:
                torch._assert_async((inputs["target_time"][task] == bounds).all(),
                                    "CoSA query group requires matching target intervals")
            state, valid = self.decoder(prepared, bounds)
            features = {name: expert(state, valid) for name, expert in self.experts.items()}
            features = self.router(state, features, valid)
            # All source experts are queried at the recipient's interval, never at
            # their independently requested output times.
            for name, tasks in self.expert_groups.items():
                selected = [task for task in tasks if task in group]
                if not selected:
                    continue
                decoded = self.readouts[name](features[name], valid, size)
                decoded = F.avg_pool2d(decoded, self.output_scale)
                outputs.update({task: self.heads[task](decoded) for task in selected})
        return self._constrain_outputs(outputs)
