"""CoSA retrieval with the existing observation-scale losses and prediction API."""

import torch
from torch import nn

from hydrostate.registry import MODELS

from ..hydrostate_model import HydroStateModel
from ..mlore.routing import MLoRERouter
from ..retrieval_model import RetrievalModelMixin
from .branches import BackgroundResidual, SpatialReadout, StateExpert
from .cgc import CGCRouter, RecurrentCGCRouter
from .encoder import CoSASharedLoRAEncoder
from .routing import StateRouter
from .state import CoSAStateFusion


@MODELS.register_module()
class CoSAHydroStateModel(RetrievalModelMixin, HydroStateModel):
    def __init__(self, expert_groups, branches=None, router=None, query_groups=None, **kwargs):
        super().__init__(**kwargs)
        from .anysat import AnySatEncoder
        if not isinstance(self.encoder, (CoSASharedLoRAEncoder, AnySatEncoder)):
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
        self._build_spatial_modules(channels, dict(router or {}))

    def _build_spatial_modules(self, channels, routing):
        self.experts = nn.ModuleDict({
            name: StateExpert(self.decoder.out_channels, channels) for name in self.expert_groups})
        self.readouts = nn.ModuleDict({
            name: SpatialReadout(channels) for name in self.expert_groups})
        kind = routing.pop("type", "StateRouter")
        router_classes = {"StateRouter": StateRouter, "CGCRouter": CGCRouter,
                          "RecurrentCGCRouter": RecurrentCGCRouter,
                          "MLoRERouter": MLoRERouter}
        if kind not in router_classes:
            raise ValueError(f"Unknown expert router: {kind}")
        self.router = router_classes[kind](
            self.expert_groups, state_channels=self.decoder.out_channels,
            channels=channels, **routing)

    def _spatial_features(self, state, valid):
        features = {name: expert(state, valid) for name, expert in self.experts.items()}
        return self.router(state, features, valid)

    def _forward(self, inputs):
        if set(inputs.get("target_time", {})) != set(self.tasks):
            raise ValueError("CoSA requires explicit target intervals for every output")
        levels = self.encoder(inputs)
        prepared = self.decoder.prepare(levels, inputs)
        size = inputs[self.encoder.input_keys[0]].shape[-2:]
        if any(length % self.output_scale for length in size):
            raise ValueError("Input dimensions must divide by output_scale")
        output_size = tuple(length // self.output_scale for length in size)
        outputs = {}
        for group in self.query_groups:
            bounds = inputs["target_time"][group[0]]
            for task in group[1:]:
                torch._assert_async((inputs["target_time"][task] == bounds).all(),
                                    "CoSA query group requires matching target intervals")
            state, valid = self.decoder(prepared, bounds)
            features = self._spatial_features(state, valid)
            # All source experts are queried at the recipient's interval, never at
            # their independently requested output times.
            for name, tasks in self.expert_groups.items():
                selected = [task for task in tasks if task in group]
                if not selected:
                    continue
                decoded = self.readouts[name](features[name], valid, output_size)
                outputs.update({task: self.heads[task](decoded) for task in selected})
        return self._constrain_outputs(outputs)


@MODELS.register_module()
class ETOnlyCGC(CoSAHydroStateModel):
    """Matched CGC control using only product ET and native station ET supervision."""

    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        terms = super()._loss_from_outputs(outputs, data_samples, auxiliary, auxiliary_weights)
        tasks = ("et_water", "et_ice", "et_land", "site_et")
        return {prefix + task: terms[prefix + task] for task in tasks
                for prefix in ("loss_", "valid_count_")}


@MODELS.register_module()
class BackgroundResidualModel(CoSAHydroStateModel):
    """Retrieval with a common local spatial block and one background per task group."""

    def _build_spatial_modules(self, channels, routing):
        self.background_residual = BackgroundResidual(
            self.expert_groups, self.decoder.out_channels, channels, **routing)
        self.readouts = nn.ModuleDict({
            name: SpatialReadout(channels) for name in self.expert_groups})

    def _spatial_features(self, state, valid):
        return self.background_residual(state, valid)


@MODELS.register_module()
class CGCSMRefinement(CoSAHydroStateModel):
    """Refit the existing SM pathway of a joint checkpoint with fixed shared features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.router, CGCRouter) or self.expert_groups.get('sm') != ('soil_moisture',):
            raise ValueError("SM refinement requires a CGC with a separate soil moisture group")
        self.requires_grad_(False)
        for module in self.sm_modules():
            module.requires_grad_(True)

    def sm_modules(self):
        return (self.experts['sm'], self.router.gates['sm'], self.readouts['sm'],
                self.heads['soil_moisture'])

    def train(self, mode=True):
        # Keep the shared representation, including stochastic layers and buffers, fixed.
        super().train(False)
        self.training = mode
        for module in self.sm_modules():
            module.train(mode)
        return self


@MODELS.register_module()
class CGCAblation(CoSAHydroStateModel):
    """One intervention at a time, keeping the original temporal state and readouts."""
    def __init__(self, intervention, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.router, CGCRouter):
            raise ValueError("Ablation requires original CGC")
        if intervention == 'constant_gate':
            # Retain each gate's learnable bias as task-specific mixture logits.
            for gate in self.router.gates.values():
                nn.init.zeros_(gate.weight)
                gate.weight.requires_grad_(False)
        elif intervention == 'pointwise':
            for expert in (*self.experts.values(), *self.router.shared):
                width = expert.spatial.first.in_channels
                # Two pointwise convolutions with 9x hidden width match the
                # kernel-weight count of the original two 3x3 convolutions.
                expert.spatial.first = nn.Conv2d(width, 9 * width, 1)
                expert.spatial.second = nn.Conv2d(9 * width, width, 1)
        else:
            raise ValueError(f"Unknown CGC intervention: {intervention}")
