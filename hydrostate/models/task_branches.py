"""Progressive task branches and simultaneous, local cross-task interaction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.normalization import TASKS
from hydrostate.registry import MODELS

from .decoder import SharedUNetDecoder


def _independent_convolutions(x, layers):
    """Run identical-layout independent convolutions on channel-packed inputs.

    Concatenation preserves the original Parameter objects and checkpoint keys;
    autograd distributes gradients back to each task's weights and bias.
    """
    first = layers[0]
    weight = torch.cat([layer.weight for layer in layers], dim=0)
    bias = None if first.bias is None else torch.cat([layer.bias for layer in layers])
    return F.conv2d(
        x, weight, bias, first.stride, first.padding, first.dilation,
        groups=sum(layer.groups for layer in layers),
    )


class TaskAttention(nn.Module):
    """SE channel gating or CBAM channel-then-spatial gating; no N x N matrix."""

    def __init__(self, channels: int, kind: str = "cbam", reduction: int = 16,
                 merge_ops: bool = True):
        super().__init__()
        if kind not in ("none", "se", "cbam"):
            raise ValueError(f"Unknown attention: {kind}")
        if reduction < 1:
            raise ValueError("attention reduction must be positive")
        self.kind = kind
        self.merge_ops = merge_ops
        if kind != "none":
            hidden = max(1, channels // reduction)
            self.channel = nn.Sequential(
                nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
            )
        if kind == "cbam":
            self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        if self.kind == "none":
            return x
        average = x.mean(dim=(-2, -1), keepdim=True)
        if self.kind == "cbam" and self.merge_ops:
            # This MLP has no batch-dependent layers: both statistics can share
            # one invocation without sharing any new parameters across tasks.
            maximum = x.amax(dim=(-2, -1), keepdim=True)
            avg_logits, max_logits = self.channel(torch.cat((average, maximum), 0)).chunk(2)
            logits = avg_logits + max_logits
        else:
            logits = self.channel(average)
            if self.kind == "cbam":
                logits = logits + self.channel(x.amax(dim=(-2, -1), keepdim=True))
        x = x * logits.sigmoid()
        if self.kind == "cbam":
            spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
            x = x * self.spatial(spatial).sigmoid()
        return x


class ResidualAdapter(nn.Module):
    """Channel bottleneck, optionally with a depthwise spatial convolution."""

    def __init__(self, channels: int, kind: str = "bottleneck", reduction: int = 4):
        super().__init__()
        if kind not in ("none", "bottleneck", "depthwise"):
            raise ValueError(f"Unknown adapter: {kind}")
        if reduction < 1:
            raise ValueError("adapter reduction must be positive")
        self.kind = kind
        if kind != "none":
            hidden = max(1, channels // reduction)
            layers = [nn.Conv2d(channels, hidden, 1), nn.GELU()]
            if kind == "depthwise":
                layers.extend([
                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU()
                ])
            layers.append(nn.Conv2d(hidden, channels, 1))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x if self.kind == "none" else x + self.layers(x)


class TaskStage(nn.Module):
    def __init__(self, channels, previous_channels, fusion, attention, adapter):
        super().__init__()
        if fusion not in ("add", "concat"):
            raise ValueError(f"Unknown stage fusion: {fusion}")
        self.fusion = fusion
        if previous_channels is not None:
            self.project = nn.Conv2d(previous_channels, channels, 1)
            if fusion == "concat":
                self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.attention = TaskAttention(channels, **attention)
        self.adapter = ResidualAdapter(channels, **adapter)

    def forward(self, shared, previous=None):
        x = shared
        if previous is not None:
            previous = self.project(previous)
            previous = F.interpolate(
                previous, size=shared.shape[-2:], mode="bilinear", align_corners=False
            )
            x = shared + previous if self.fusion == "add" else self.fuse(
                torch.cat((shared, previous), dim=1)
            )
        return self.adapter(self.attention(x))


@MODELS.register_module()
class StagedTaskDecoder(SharedUNetDecoder):
    """One shared trunk, with an independent progressive branch for each task."""

    def __init__(
        self,
        in_channels=768,
        channels=(512, 256, 128),
        scale_factor=4,
        fusion="add",
        attention=None,
        adapter=None,
        merge_ops=True,
    ):
        super().__init__(in_channels=in_channels, channels=channels, scale_factor=scale_factor)
        self.merge_ops = merge_ops
        attention = dict(kind="cbam", reduction=16) if attention is None else dict(attention)
        attention.setdefault("merge_ops", merge_ops)
        adapter = dict(kind="bottleneck", reduction=4) if adapter is None else adapter
        self.branches = nn.ModuleDict({
            task: nn.ModuleList([
                TaskStage(c, channels[i - 1] if i else None, fusion, attention, adapter)
                for i, c in enumerate(channels)
            ])
            for task in TASKS
        })

    def forward(self, features):
        shared = self.forward_stages(features)
        if self.merge_ops:
            return self._forward_merged(shared)
        outputs = {}
        for task, stages in self.branches.items():
            x = None
            for stage, shared_feature in zip(stages, shared):
                x = stage(shared_feature, x)
            outputs[task] = x
        return outputs

    def _forward_merged(self, shared):
        previous = None
        for index, shared_feature in enumerate(shared):
            stages = [self.branches[task][index] for task in TASKS]
            if previous is None:
                inputs = [shared_feature] * len(stages)
            else:
                projected = _independent_convolutions(
                    torch.cat(previous, 1), [stage.project for stage in stages]
                )
                projected = F.interpolate(
                    projected, size=shared_feature.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).chunk(len(stages), dim=1)
                if stages[0].fusion == "add":
                    inputs = [shared_feature + value for value in projected]
                else:
                    # Interleave shared/previous channels *within* each task.
                    packed = torch.cat([
                        value for previous_task in projected
                        for value in (shared_feature, previous_task)
                    ], dim=1)
                    inputs = _independent_convolutions(
                        packed, [stage.fuse for stage in stages]
                    ).chunk(len(stages), dim=1)
            inputs = [stage.attention(value) for stage, value in zip(stages, inputs)]
            if stages[0].adapter.kind == "none":
                previous = inputs
                continue
            residual = torch.cat(inputs, dim=1)
            value = residual
            for layer_index, layer in enumerate(stages[0].adapter.layers):
                if isinstance(layer, nn.Conv2d):
                    value = _independent_convolutions(
                        value, [stage.adapter.layers[layer_index] for stage in stages]
                    )
                else:
                    value = layer(value)  # GELU is elementwise and has no parameters.
            previous = (residual + value).chunk(len(stages), dim=1)
        return dict(zip(TASKS, previous))


@MODELS.register_module()
class TaskGatedInteraction(nn.Module):
    """One simultaneous round of six directed messages at the output grid.

    Each edge has a 1x1 value projection and a scalar spatial gate computed
    from concatenated recipient/source features. Sigmoid messages are averaged;
    softmax weights sum to one over the two incoming sources at each pixel.
    """

    def __init__(self, channels=128, mode="sigmoid", residual_scale=0.1, merge_ops=True,
                 learnable_scale=False):
        super().__init__()
        if mode not in ("none", "sigmoid", "softmax"):
            raise ValueError(f"Unknown interaction mode: {mode}")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.mode = mode
        self.residual_scale = residual_scale
        self.merge_ops = merge_ops
        self.messages = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.task_scales = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(float(residual_scale))) for task in TASKS
        }) if learnable_scale and mode != "none" else nn.ParameterDict()
        if mode != "none":
            for target in TASKS:
                for source in TASKS:
                    if source == target:
                        continue
                    key = f"{source}_to_{target}"
                    self.messages[key] = nn.Conv2d(channels, channels, 1, bias=False)
                    self.gates[key] = nn.Conv2d(2 * channels, 1, 1)

    def forward(self, features):
        if self.mode == "none":
            return dict(features)
        projected = {}
        if self.merge_ops:
            for source in TASKS:
                keys = [f"{source}_to_{target}" for target in TASKS if target != source]
                # Each source still has two distinct outgoing weight matrices.
                weight = torch.cat([self.messages[key].weight for key in keys], dim=0)
                values = F.conv2d(features[source], weight).chunk(len(keys), dim=1)
                projected.update(zip(keys, values))
        outputs = {}
        for target in TASKS:
            messages, logits = [], []
            for source in TASKS:
                if source == target:
                    continue
                key = f"{source}_to_{target}"
                messages.append(
                    projected[key] if self.merge_ops else self.messages[key](features[source])
                )
                logits.append(self.gates[key](torch.cat((features[target], features[source]), 1)))
            logits = torch.stack(logits, dim=1)  # B,2,1,H,W
            weights = logits.softmax(dim=1) if self.mode == "softmax" else logits.sigmoid() / 2
            if self.merge_ops:
                update = weights[:, 0] * messages[0] + weights[:, 1] * messages[1]
            else:
                update = (weights * torch.stack(messages, dim=1)).sum(dim=1)
            scale = self.task_scales[target] if target in self.task_scales else self.residual_scale
            outputs[target] = features[target] + scale * update
        return outputs
