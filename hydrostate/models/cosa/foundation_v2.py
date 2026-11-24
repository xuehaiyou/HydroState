"""Multi-stage OlmoEarth adapters with interval-conditioned paper-style decoders."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import StochasticDepth

from hydrostate.registry import MODELS

from ..heads import activate_prediction
from ..multilevel_encoder import multilevel_features
from ..multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from .d2moe import D2MoEStateFusion, adapt_mixture_block
from .ditask import AdapterDropout, DiTASKStateFusion, adapt_spectral_block
from .encoder import CoSASharedLoRAEncoder
from .faar import FAARDecoder, RankedDirection, adapt_rank_block
from .model import CoSAHydroStateModel
from .nexusflow import NexusFlow
from .rutal import ResidualExpertFFN, RUTaLDecoder

GROUPS = ("coverage", "sm", "et")


@MODELS.register_module()
class FoundationStageEncoderV2(CoSASharedLoRAEncoder):
    def __init__(self, method, gradient_checkpointing=False, **kwargs):
        for key in (
            "shared_rank",
            "private_rank",
            "lora_scale",
            "train_layer_norm",
            "train_patch_embedding",
        ):
            kwargs.pop(key, None)
        MultiLevelTemporalOlmoEarthEncoder.__init__(self, gradient_checkpointing=False, **kwargs)
        self.finetuning = "lora"  # Reuse MTLoRA's grouped depth restoration.
        self.method = method
        if method not in ("ditask", "d2moe", "faar", "rutal"):
            raise ValueError(method)
        self.mixture = method == "d2moe"
        self.tasks = (*GROUPS, "prior") if self.mixture else GROUPS
        core = self.backbone.model.requires_grad_(False)
        if method in ("ditask", "faar"):
            for module in core.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
            for key in ("sentinel1", "sentinel2_l2a", "landsat"):
                core.patch_embeddings.per_modality_embeddings[key].requires_grad_(True)
        adapters = dict(
            d2moe=adapt_mixture_block, ditask=adapt_spectral_block, faar=adapt_rank_block
        )
        for i, block in enumerate(core.blocks):
            stage = i in self.out_indices
            if method == "ditask":
                # Swin-T reference increases the residual drop probability linearly to 0.2.
                block.drop_path = StochasticDepth(0.2 * i / max(len(core.blocks) - 1, 1), "row")
            if method == "rutal":
                block.mlp = ResidualExpertFFN(block.mlp)
            else:
                block = adapters[method](block, stage, GROUPS)
            core.blocks[i] = block
            if gradient_checkpointing:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        self.train(self.training)

    def _extract_levels(self, sample, out_indices):
        if self.method == "rutal":
            return {
                "shared": multilevel_features(
                    self.backbone.model,
                    sample,
                    self.patch_size,
                    out_indices,
                    retain_tokens=True,
                    include_patch=True,
                )
            }
        return super()._extract_levels(sample, out_indices)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, (AdapterDropout, StochasticDepth, RankedDirection)):
                module.train(mode)
        return self

    def forward(self, inputs):
        return self.forward_levels(inputs, self.out_indices)


@MODELS.register_module()
class FoundationStageModelV2(CoSAHydroStateModel):
    def _build_spatial_modules(self, channels, routing):
        expected = dict(
            d2moe=D2MoEStateFusion, ditask=DiTASKStateFusion, faar=FAARDecoder, rutal=RUTaLDecoder
        )[self.encoder.method]
        if not isinstance(self.decoder, expected) or routing.get("enabled", False):
            raise ValueError("Stage adapters require their own decoder with CoSA routing disabled")
        self.activations = {task: head.activation for task, head in self.heads.items()}
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(self.decoder.head_channels, len(tasks), 1)
                for name, tasks in self.expert_groups.items()
            }
        )

    def _forward(self, inputs):
        return self._run_stages(inputs)[0]

    def _run_stages(self, inputs, alignment=None):
        if set(inputs.get("target_time", {})) != set(self.tasks):
            raise ValueError("Every output requires an explicit target interval")
        levels = self.encoder(inputs)
        prepared = {name: self.decoder.prepare(seq, inputs) for name, seq in levels.items()}
        size = inputs[self.encoder.input_keys[0]].shape[-2:]
        if any(length % self.output_scale for length in size):
            raise ValueError("Input grid must divide by output_scale")
        output_size = tuple(n // self.output_scale for n in size)
        outputs, auxiliary = {}, []
        for query in self.query_groups:
            bounds = inputs["target_time"][query[0]]
            for task in query[1:]:
                torch._assert_async(
                    (inputs["target_time"][task] == bounds).all(), "Mismatched query interval"
                )
            groups = [
                g for g, members in self.expert_groups.items() if set(members).intersection(query)
            ]
            if alignment is not None:
                groups = tuple(self.expert_groups)
            if self.encoder.mixture:
                prior = self.decoder.prior_map(prepared["prior"], bounds)
            if self.encoder.method in ("faar", "rutal"):
                states, valid = self.decoder.states(prepared, bounds, groups)
            else:
                states = {}
                for name in groups:
                    if self.encoder.mixture:
                        states[name], valid = self.decoder(
                            prepared[name], bounds, task=name, prior=prior
                        )
                    else:
                        states[name], valid = self.decoder(prepared[name], bounds, task=name)
            if alignment is not None:
                auxiliary.append(alignment(states, valid))
            for name, state in states.items():
                tasks = self.expert_groups[name]
                selected = [t for t in tasks if t in query]
                logits = self.heads[name](state.masked_fill(~valid, 0))
                logits = F.interpolate(
                    logits, size=output_size, mode="bilinear", align_corners=False
                )
                for task, value in zip(tasks, logits.split(1, dim=1)):
                    if task in selected:
                        outputs[task] = activate_prediction(value, self.activations[task])
        return self._constrain_outputs(outputs), auxiliary


@MODELS.register_module()
class NexusD2MoE(FoundationStageModelV2):
    def __init__(self, alignment_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        if not self.encoder.mixture:
            raise ValueError("NexusD2MoE requires the D2MoE stage encoder")
        self.alignment = NexusFlow(self.decoder.head_channels, self.expert_groups)
        self.alignment_weight = alignment_weight

    def loss(self, inputs, data_samples):
        outputs, aligned = self._run_stages(inputs, self.alignment)
        terms = self._loss_from_outputs(outputs, self._land_samples(inputs, data_samples))
        terms["loss_nexus_alignment"] = self.alignment_weight * torch.stack(aligned).mean()
        return terms
