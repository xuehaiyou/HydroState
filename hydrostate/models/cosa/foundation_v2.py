"""Multi-stage OlmoEarth adapters with interval-conditioned paper-style decoders."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import StochasticDepth

from hydrostate.registry import MODELS

from ..heads import activate_prediction
from ..multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from .d2moe import D2MoEStateFusion, adapt_mixture_block
from .ditask import AdapterDropout, DiTASKStateFusion, adapt_spectral_block
from .encoder import CoSASharedLoRAEncoder
from .model import CoSAHydroStateModel

GROUPS = ("coverage", "sm", "et")


@MODELS.register_module()
class FoundationStageEncoderV2(CoSASharedLoRAEncoder):
    def __init__(self, method, gradient_checkpointing=False, **kwargs):
        for key in ("shared_rank", "private_rank", "lora_scale", "train_layer_norm",
                    "train_patch_embedding"):
            kwargs.pop(key, None)
        MultiLevelTemporalOlmoEarthEncoder.__init__(self, gradient_checkpointing=False, **kwargs)
        self.finetuning = "lora"  # Reuse MTLoRA's grouped depth restoration.
        if method not in ("ditask", "d2moe"):
            raise ValueError(method)
        self.mixture = method == "d2moe"
        self.tasks = (*GROUPS, "prior") if self.mixture else GROUPS
        core = self.backbone.model.requires_grad_(False)
        if not self.mixture:
            for module in core.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
            for key in ("sentinel1", "sentinel2_l2a", "landsat"):
                core.patch_embeddings.per_modality_embeddings[key].requires_grad_(True)
        adapt = adapt_mixture_block if self.mixture else adapt_spectral_block
        for i, block in enumerate(core.blocks):
            stage = i in self.out_indices
            if not self.mixture:
                # Swin-T reference increases the residual drop probability linearly to 0.2.
                block.drop_path = StochasticDepth(0.2 * i / max(len(core.blocks) - 1, 1), "row")
            block = adapt(block, stage, GROUPS)
            core.blocks[i] = block
            if gradient_checkpointing:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, (AdapterDropout, StochasticDepth)):
                module.train(mode)
        return self

    def forward(self, inputs):
        return self.forward_levels(inputs, self.out_indices)


@MODELS.register_module()
class FoundationStageModelV2(CoSAHydroStateModel):
    def _build_spatial_modules(self, channels, routing):
        expected = D2MoEStateFusion if self.encoder.mixture else DiTASKStateFusion
        if not isinstance(self.decoder, expected) or routing.get("enabled", False):
            raise ValueError("Stage adapters require their own decoder with CoSA routing disabled")
        self.activations = {task: head.activation for task, head in self.heads.items()}
        self.heads = nn.ModuleDict({name: nn.Conv2d(self.decoder.head_channels, len(tasks), 1)
                                    for name, tasks in self.expert_groups.items()})

    def _forward(self, inputs):
        if set(inputs.get("target_time", {})) != set(self.tasks):
            raise ValueError("Every output requires an explicit target interval")
        levels = self.encoder(inputs)
        prepared = {name: self.decoder.prepare(seq, inputs) for name, seq in levels.items()}
        size = inputs[self.encoder.input_keys[0]].shape[-2:]
        if any(length % self.output_scale for length in size):
            raise ValueError("Input grid must divide by output_scale")
        output_size = tuple(n // self.output_scale for n in size)
        outputs = {}
        for query in self.query_groups:
            bounds = inputs["target_time"][query[0]]
            for task in query[1:]:
                torch._assert_async((inputs["target_time"][task] == bounds).all(),
                                    "Mismatched query interval")
            if self.encoder.mixture:
                prior = self.decoder.prior_map(prepared["prior"], bounds)
            for name, tasks in self.expert_groups.items():
                selected = [t for t in tasks if t in query]
                if not selected:
                    continue
                if self.encoder.mixture:
                    state, valid = self.decoder(prepared[name], bounds, task=name, prior=prior)
                else:
                    state, valid = self.decoder(prepared[name], bounds, task=name)
                logits = self.heads[name](state.masked_fill(~valid, 0))
                logits = F.interpolate(logits, size=output_size, mode="bilinear",
                                       align_corners=False)
                for task, value in zip(tasks, logits.split(1, dim=1)):
                    if task in selected:
                        outputs[task] = activate_prediction(value, self.activations[task])
        return self._constrain_outputs(outputs)
