"""OlmoEarth stage adapters paired with the independent B readouts."""

from contextvars import ContextVar
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hydrostate.registry import MODELS

from ..multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from ..shared_private_lora import LowRankUpdate
from .encoder import CoSASharedLoRAEncoder
from .model import CoSAHydroStateModel

GROUPS = ("coverage", "sm", "et")
_ROUTE = ContextVar("foundation_route", default=(None, None))


def interpolate(values, position):
    index = position.clamp(0, 1) * (values.numel() - 1)
    left = index.long()
    return torch.lerp(values[left], values[(left + 1).clamp_max(values.numel() - 1)],
                      index - left)


class SpectralProjection(nn.Module):
    """Warp the sorted singular spectrum by a boundary-fixed piecewise affine flow."""

    def __init__(self, base, stage, cells=16):
        super().__init__()
        u, s, vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
        self.register_buffer("u", u.contiguous())
        self.register_buffer("s", s)
        self.register_buffer("vh", vh.contiguous())
        self.register_buffer("grid", torch.linspace(0, 1, s.numel()))
        self.bias = base.bias
        names = ("shared", *GROUPS) if stage else ("shared",)
        self.velocity = nn.ParameterDict({k: nn.Parameter(torch.zeros(cells - 1)) for k in names})

    def forward(self, x):
        group, _ = _ROUTE.get()
        velocity = self.velocity[group if group in GROUPS else "shared"]
        # Small monotone Euler step followed by scaling and squaring integrates
        # a continuous piecewise-affine velocity; endpoints stay fixed.
        nodes = F.pad(velocity.tanh(), (1, 1))
        warp = self.grid + interpolate(nodes, self.grid) / (4 * nodes.numel())
        for _ in range(5):
            warp = interpolate(warp, warp)
        singular = interpolate(self.s, warp)
        weight = (self.u * singular) @ self.vh
        return F.linear(x, weight, self.bias)


class MixtureProjection(nn.Module):
    def __init__(self, base, stage):
        super().__init__()
        self.base = base
        dims = base.in_features, base.out_features
        self.shared = LowRankUpdate(*dims, 16, scale=4)
        self.experts = nn.ModuleList(LowRankUpdate(*dims, 4, scale=4)
                                     for _ in range(4 if stage else 0))
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        task, weights = _ROUTE.get()
        value = self.base(x)
        if task == "prior":
            return value
        inputs = self.dropout(x)
        value = value + self.shared(inputs)
        if weights is not None:
            for index, expert in enumerate(self.experts):
                value = value + expert(inputs) * weights[:, None, index:index + 1].to(value.dtype)
        return value


class StageExit(nn.Module):
    def __init__(self, block, mixture):
        super().__init__()
        self.block = block
        self.mixture = mixture
        if mixture:
            self.gates = nn.ModuleDict({name: nn.Linear(block.norm1.weight.numel(), 4)
                                        for name in GROUPS})

    def forward(self, x, **kwargs):
        shared = self.block(x=x, **kwargs)
        if _ROUTE.get()[0] == "prior":
            return (shared, shared, shared, shared)
        summary = None
        if self.mixture:
            valid = kwargs["attn_mask"]
            summary = (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        branches = []
        for task in GROUPS:
            weights = None
            if self.mixture:
                scores, indices = self.gates[task](summary).topk(2)
                probabilities = scores.float().softmax(-1)
                weights = probabilities.new_zeros(len(x), 4).scatter(1, indices, probabilities)
            token = _ROUTE.set((task, weights))
            try:
                branches.append(self.block(x=x, **kwargs))
            finally:
                _ROUTE.reset(token)
        return (shared, *branches)


@MODELS.register_module()
class FoundationStageEncoder(CoSASharedLoRAEncoder):
    def __init__(self, method, gradient_checkpointing=False, **kwargs):
        for key in ("shared_rank", "private_rank", "lora_scale", "train_layer_norm",
                    "train_patch_embedding"):
            kwargs.pop(key, None)
        MultiLevelTemporalOlmoEarthEncoder.__init__(self, gradient_checkpointing=False, **kwargs)
        self.tasks = GROUPS
        self.finetuning = "lora"  # Reuse MTLoRA's grouped depth restoration.
        if method not in ("ditask", "d2moe"):
            raise ValueError(method)
        self.mixture = method == "d2moe"
        core = self.backbone.model.requires_grad_(False)
        if not self.mixture:
            for module in core.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
            for key in ("sentinel1", "sentinel2_l2a", "landsat"):
                core.patch_embeddings.per_modality_embeddings[key].requires_grad_(True)
        for i, block in enumerate(core.blocks):
            stage = i in self.out_indices
            cls = MixtureProjection if self.mixture else SpectralProjection
            for name in ("q", "k", "v", "proj"):
                setattr(block.attn, name, cls(getattr(block.attn, name), stage))
            if stage:
                block = StageExit(block, self.mixture)
                core.blocks[i] = block
            if gradient_checkpointing:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        if self.mixture:
            self.prior_gate = nn.ModuleDict({name: nn.Linear(self.out_channels, self.out_channels)
                                             for name in GROUPS})
        self.backbone.eval()

    def forward(self, inputs):
        result = self.forward_levels(inputs, self.out_indices)
        if self.mixture:
            ctx = _ROUTE.set(("prior", None))
            try:
                with torch.no_grad():
                    prior = self.forward_levels(inputs, self.out_indices)[GROUPS[0]][-1]["features"]
            finally:
                _ROUTE.reset(ctx)
            for name in GROUPS:
                for level in result[name]:
                    value = level["features"]
                    gate = self.prior_gate[name](value.movedim(2, -1)).sigmoid().movedim(-1, 2)
                    level["features"] = value + gate * prior
        return result


@MODELS.register_module()
class FoundationStageModel(CoSAHydroStateModel):
    def _forward(self, inputs):
        levels = self.encoder(inputs)
        size = inputs["s1"].shape[-2:]
        output_size = tuple(n // self.output_scale for n in size)
        outputs = {}
        for name, tasks in self.expert_groups.items():
            prepared = self.decoder.prepare(levels[name], inputs)
            for query in self.query_groups:
                selected = [t for t in tasks if t in query]
                if selected:
                    bounds = inputs["target_time"][query[0]]
                    for task in selected:
                        torch._assert_async((inputs["target_time"][task] == bounds).all(),
                                            "Mismatched query interval")
                    state, valid = self.decoder(prepared, bounds)
                    feature = self.experts[name](state, valid)
                    decoded = self.readouts[name](feature, valid, output_size)
                    outputs.update({task: self.heads[task](decoded) for task in selected})
        return self._constrain_outputs(outputs)
