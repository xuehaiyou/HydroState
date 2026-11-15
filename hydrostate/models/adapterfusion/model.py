"""Frozen shared-LoRA experts with trainable official AdapterFusion residuals."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from ..mtlora import MTLoRAHydroStateModel
from .fusion import ResidualAdapterFusion


@MODELS.register_module()
class AdapterFusionHydroStateModel(MTLoRAHydroStateModel):
    def __init__(self, baseline_checkpoint=None, fusion_enabled=True, **kwargs):
        if kwargs.get("encoder", {}).get("private_rank", 0) != 0:
            raise ValueError("AdapterFusion experiment requires shared-LoRA (private_rank=0)")
        super().__init__(**kwargs)
        # Strictly load the original model BEFORE adding new parameter names.
        if baseline_checkpoint:
            state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
            self.load_state_dict(state["state_dict"], strict=True)
        self.requires_grad_(False)
        self.fusions = nn.ModuleDict({
            task: ResidualAdapterFusion(self.decoders[task].out_channels) for task in self.tasks})
        self.fusion_enabled = fusion_enabled
        self.train(False)

    def train(self, mode=True):
        # eval() and frozen parameters are separate requirements: enforce both.
        super().train(False)
        self.training = mode
        if hasattr(self, "fusions"):
            self.fusions.train(mode)
        return self

    @torch.no_grad()
    def _experts(self, levels, inputs, bounds, auxiliary_cache):
        features, validity = {}, {}
        for task in self.tasks:
            query_name = next(group[0] for group in self.query_groups if task in group)
            values = tuple(self.queries[task](level, bounds, query_name) for level in levels[task])
            if self.input_fusion is not None:
                values = tuple(
                    self.input_fusion(value, inputs, cache=auxiliary_cache) for value in values)
            features[task] = self.decoders[task].fuse_levels(values)
            validity[task] = torch.stack([level["valid"].any(1) for level in levels[task]]).any(0)
        return features, validity

    def _forward(self, inputs):
        if set(inputs.get("target_time", {})) != set(self.tasks):
            raise ValueError("Explicit target bounds required for every task")
        for group in self.query_groups:
            for task in group[1:]:
                torch._assert_async(
                    (inputs["target_time"][task] == inputs["target_time"][group[0]]).all(),
                                    "Joint fractions must share a target interval")
        if not self.fusion_enabled:
            return super()._forward(inputs)
        with torch.no_grad():
            levels = self.encoder(inputs)
        outputs, cache, auxiliary_cache = {}, [], {}
        for task in self.tasks:
            bounds = inputs["target_time"][task]
            match = next((value for prior, value in cache if torch.equal(prior, bounds)), None)
            if match is None:
                match = self._experts(levels, inputs, bounds, auxiliary_cache)
                cache.append((bounds, match))
            features, valid = match
            value = self.fusions[task](features[task], torch.stack(list(features.values()), 1),
                                       torch.stack(list(valid.values()), 1), valid[task])
            # Frozen decoder weights still propagate gradients to the fusion.
            decoded = self.decoders[task].decode_spatial(value)
            if any(size % self.output_scale for size in decoded.shape[-2:]):
                raise ValueError("Decoded grid must divide by output_scale")
            outputs[task] = self.heads[task](F.avg_pool2d(decoded, self.output_scale))
        return self._constrain_outputs(outputs)
