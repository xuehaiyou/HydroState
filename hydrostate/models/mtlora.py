"""MTLoRA-style stage exits for OlmoEarth, with one shared attention traversal.

Stages here are selected encoder depths, not Swin spatial scales. TS branches
use W + private BA; the continuing stream uses W + shared BA (paper Eq. 2).
"""

import copy
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hydrostate.registry import MODELS
from hydrostate.tasks import resolve_tasks, task_config

from .multilevel_encoder import multilevel_features
from .multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from .olmoearth_encoder import MODALITIES
from .retrieval_model import RetrievalModelMixin
from .shared_private_lora import LowRankUpdate
from .temporal_model import TemporalHydroStateModel


class MTLoRALinear(nn.Module):
    """Frozen pretrained map and separate shared/private low-rank updates."""

    def __init__(self, base, shared_rank, private_rank=0, tasks=(), scale=1.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("MTLoRA requires a linear projection")
        if type(private_rank) is not int or private_rank < 0:
            raise ValueError("private_rank must be a nonnegative integer")
        self.base = base.requires_grad_(False)
        self.shared = LowRankUpdate(base.in_features, base.out_features, shared_rank, scale)
        self.private = nn.ModuleDict(
            {
                task: LowRankUpdate(base.in_features, base.out_features, private_rank, scale)
                for task in tasks
                if private_rank
            }
        )

    def forward(self, x, task=None):
        update = self.shared if task is None or not self.private else self.private[task]
        return self.base(x) + update(x)


class MTLoRAExitBlock(nn.Module):
    """Compute attention once; branch only its output projection and the MLP."""

    def __init__(self, block, tasks, shared_rank, private_rank, scale):
        super().__init__()
        self.tasks = tuple(tasks)
        self.private_rank = private_rank
        self.projection = MTLoRALinear(block.attn.proj, shared_rank, private_rank, tasks, scale)
        # Preserve upstream QKV, RoPE, masking and SDPA; extract unprojected attention.
        block.attn.proj = nn.Identity()
        self.proj_drop = block.attn.proj_drop
        block.attn.proj_drop = nn.Identity()
        self.attn = block.attn
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.ls1, self.ls2, self.drop_path = block.ls1, block.ls2, block.drop_path
        self.fc1 = MTLoRALinear(block.mlp.fc1, shared_rank, private_rank, tasks, scale)
        self.fc2 = MTLoRALinear(block.mlp.fc2, shared_rank, private_rank, tasks, scale)
        self.act, self.drop1, self.drop2 = block.mlp.act, block.mlp.drop1, block.mlp.drop2

    def branch(self, x, attended, task=None):
        value = x + self.drop_path(self.ls1(self.proj_drop(self.projection(attended, task))))
        hidden = self.drop1(self.act(self.fc1(self.norm2(value), task)))
        return value + self.drop_path(self.ls2(self.drop2(self.fc2(hidden, task))))

    def forward(self, x, **kwargs):
        attended = self.attn(x=self.norm1(x), **kwargs)
        shared = self.branch(x, attended)
        branches = (
            tuple(self.branch(x, attended, task) for task in self.tasks)
            if self.private_rank
            else (shared,) * len(self.tasks)
        )
        return (shared, *branches)


@MODELS.register_module()
class MTLoRAOlmoEarthEncoder(MultiLevelTemporalOlmoEarthEncoder):
    """Depth exits with LoRA adaptation or a full-backbone training control."""

    def __init__(
        self,
        tasks=None,
        shared_rank=8,
        private_rank=4,
        lora_scale=1.0,
        train_patch_embedding=True,
        train_layer_norm=True,
        gradient_checkpointing=False,
        finetuning="lora",
        **kwargs,
    ):
        if finetuning not in ("lora", "full"):
            raise ValueError("finetuning must be 'lora' or 'full'")
        if finetuning == "full" and (shared_rank != 0 or private_rank != 0):
            raise ValueError("Full fine-tuning requires shared_rank=private_rank=0")
        super().__init__(gradient_checkpointing=False, **kwargs)
        self.tasks = tuple(resolve_tasks(tasks))
        self.finetuning = finetuning
        core = self.backbone.model
        if finetuning == "lora":
            core.requires_grad_(False)
            if train_patch_embedding:
                for name in MODALITIES:
                    core.patch_embeddings.per_modality_embeddings[name].requires_grad_(True)
            if train_layer_norm:
                for module in core.modules():
                    if isinstance(module, nn.LayerNorm):
                        module.requires_grad_(True)
        else:
            # Preserve upstream trainability, including fixed encodings. The
            # unused modalities and pretraining projector have no retrieval loss.
            for name, embedding in core.patch_embeddings.per_modality_embeddings.items():
                if name not in MODALITIES or not train_patch_embedding:
                    embedding.requires_grad_(False)
            if not train_layer_norm:
                for module in core.modules():
                    if isinstance(module, nn.LayerNorm):
                        module.requires_grad_(False)
        # Never unfreeze the unused pretraining projection's normalization layers.
        core.project_and_aggregate.requires_grad_(False)
        for index, block in enumerate(core.blocks):
            if finetuning == "lora":
                for name in ("q", "k", "v"):
                    setattr(
                        block.attn,
                        name,
                        MTLoRALinear(getattr(block.attn, name), shared_rank, scale=lora_scale),
                    )
                if index in self.out_indices:
                    block = MTLoRAExitBlock(
                        block, self.tasks, shared_rank, private_rank, lora_scale
                    )
                    core.blocks[index] = block
                else:
                    block.attn.proj = MTLoRALinear(block.attn.proj, shared_rank, scale=lora_scale)
                    block.mlp.fc1 = MTLoRALinear(block.mlp.fc1, shared_rank, scale=lora_scale)
                    block.mlp.fc2 = MTLoRALinear(block.mlp.fc2, shared_rank, scale=lora_scale)
            if gradient_checkpointing:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        # Backbone stochastic masking/dropout stay deterministic during adaptation.
        self.backbone.eval()
        if private_rank:
            # The last shared output has no consumer; avoid trainable dead weights.
            last = core.blocks[-1]
            for layer in (last.projection, last.fc1, last.fc2):
                layer.shared.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_levels(self, sample, out_indices):
        if self.finetuning == "full":
            levels = super()._extract_levels(sample, out_indices)
            return {task: levels for task in self.tasks}
        return multilevel_features(
            self.backbone.model,
            sample,
            self.patch_size,
            out_indices,
            retain_tokens=True,
            branch_tasks=self.tasks,
        )


@MODELS.register_module()
class MTLoRAFusionDecoder(nn.Module):
    """Same-grid depth projection, concatenation and residual fusion, then upsample."""

    def __init__(self, in_channels=768, channels=64, num_levels=3, scale_factor=4):
        super().__init__()
        if min(in_channels, channels, num_levels, scale_factor) <= 0:
            raise ValueError("Fusion dimensions must be positive")
        self.out_channels, self.scale_factor = channels, scale_factor
        self.projections = nn.ModuleList(
            nn.Conv2d(in_channels, channels, 1) for _ in range(num_levels)
        )
        self.compress = nn.Conv2d(channels * num_levels, channels, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, levels):
        if len(levels) != len(self.projections) or any(x.shape != levels[0].shape for x in levels):
            raise ValueError("Fusion requires equal-grid depth features")
        value = self.compress(torch.cat([p(x) for p, x in zip(self.projections, levels)], dim=1))
        value = value + self.residual(value)
        value = F.interpolate(
            value, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        return self.refine(value)


@MODELS.register_module()
class MTLoRAHydroStateModel(RetrievalModelMixin, TemporalHydroStateModel):
    """One encoder traversal with independent task queries, fusion and heads."""

    def __init__(self, **kwargs):
        specs = resolve_tasks(kwargs.get("tasks"))
        kwargs["tasks"] = task_config(specs)
        encoder = dict(kwargs["encoder"])
        if "tasks" in encoder and resolve_tasks(encoder["tasks"]) != specs:
            raise ValueError("Encoder tasks disagree with model tasks")
        encoder["tasks"] = task_config(specs)
        kwargs["encoder"] = encoder
        super().__init__(**kwargs)
        if not isinstance(self.encoder, MTLoRAOlmoEarthEncoder):
            raise TypeError("MTLoRAHydroStateModel requires MTLoRAOlmoEarthEncoder")
        if not isinstance(self.decoder, MTLoRAFusionDecoder):
            raise TypeError("MTLoRAHydroStateModel requires MTLoRAFusionDecoder")
        template = self.decoder
        del self.decoder
        self.decoders = nn.ModuleDict({task: copy.deepcopy(template) for task in self.tasks})
        template = self.query_time
        del self.query_time
        self.queries = nn.ModuleDict({task: copy.deepcopy(template) for task in self.tasks})

    def _forward(self, inputs):
        if "target_time" not in inputs or set(inputs["target_time"]) != set(self.tasks):
            raise ValueError("Explicit query bounds are required for every task")
        for group in self.query_groups:
            for task in group[1:]:
                torch._assert_async(
                    (inputs["target_time"][task] == inputs["target_time"][group[0]]).all(),
                    "Joint fractions must share query bounds",
                )
        levels = self.encoder(inputs)
        outputs = {}
        auxiliary_cache = {}
        for task in self.tasks:
            query_name = next(group[0] for group in self.query_groups if task in group)
            features = tuple(
                self.queries[task](level, inputs["target_time"][task], query_name)
                for level in levels[task]
            )
            if self.input_fusion is not None:
                features = tuple(self.input_fusion(value, inputs, cache=auxiliary_cache)
                                 for value in features)
            decoded = self.decoders[task](features)
            if any(size % self.output_scale for size in decoded.shape[-2:]):
                raise ValueError("Decoded spatial dimensions must divide by output_scale")
            outputs[task] = self.heads[task](F.avg_pool2d(decoded, self.output_scale))
        return self._constrain_outputs(outputs)
