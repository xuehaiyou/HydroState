"""Shared multilevel fusion followed by output-grid task adaptation."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .decoder import ConvBlock
from .hydrostate_model import HydroStateModel


@MODELS.register_module()
class MultiLevelFusionDecoder(nn.Module):
    """Deep-to-shallow additive fusion; all encoder levels share a spatial grid."""

    def __init__(self, in_channels=768, channels=(256, 128, 64)):
        super().__init__()
        if len(channels) != 3 or min(channels) <= 0 or in_channels <= 0:
            raise ValueError("Multilevel decoder requires three positive channel widths")
        self.in_channels = in_channels
        self.laterals = nn.ModuleList(nn.Conv2d(in_channels, c, 1) for c in channels)
        self.transitions = nn.ModuleList(
            nn.Conv2d(a, b, 1) for a, b in zip(channels[:-1], channels[1:])
        )
        self.blocks = nn.ModuleList(ConvBlock(c, c) for c in channels)
        self.out_channels = channels[-1]

    def forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise ValueError("Multilevel decoder requires three shallow-to-deep feature maps")
        shape = features[0].shape
        if len(shape) != 4 or shape[1] != self.in_channels:
            raise ValueError("Invalid encoder feature shape")
        if any(x.shape != shape for x in features):
            raise ValueError("OlmoEarth levels must share batch, channels and spatial grid")
        x = self.blocks[0](self.laterals[0](features[-1]))
        for index, feature in enumerate(reversed(features[:-1]), start=1):
            # Reduce channels before upsampling to avoid high-resolution wide convolutions.
            x = F.interpolate(self.transitions[index - 1](x), scale_factor=2,
                              mode="bilinear", align_corners=False)
            lateral = F.interpolate(self.laterals[index](feature), size=x.shape[-2:],
                                    mode="bilinear", align_corners=False)
            x = self.blocks[index](x + lateral)
        return x


class OutputTaskAdapter(nn.Module):
    """64 -> 16 -> 16 -> 64 residual adjustment, initially an identity."""

    def __init__(self, channels=64, bottleneck_channels=16, enabled=True):
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("Adapter widths must be positive")
        self.enabled = enabled
        self.layers = nn.Sequential()
        if enabled:
            self.layers = nn.Sequential(
                nn.Conv2d(channels, bottleneck_channels, 1), nn.GELU(),
                nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(bottleneck_channels, channels, 1),
            )
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return features + self.layers(features) if self.enabled else features


@MODELS.register_module()
class MultiLevelHydroStateModel(HydroStateModel):
    """No auxiliary heads; reuse the common loss, WF-SM support and predict API."""

    def __init__(self, adapter=None, interaction=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.decoder, MultiLevelFusionDecoder):
            raise ValueError("MultiLevelHydroStateModel requires MultiLevelFusionDecoder")
        channels = self.decoder.out_channels
        self.adapters = nn.ModuleDict({
            task: OutputTaskAdapter(channels=channels, **dict(adapter or {})) for task in self.tasks
        })
        cfg = dict(interaction or {})
        cfg.setdefault("type", "TaskGatedInteraction")
        cfg.setdefault("tasks", self.task_specs)
        cfg.setdefault("channels", channels)
        cfg.setdefault("learnable_scale", True)
        if cfg["channels"] != channels:
            raise ValueError("Interaction channels must match decoder output")
        self.interaction = MODELS.build(cfg)

    def _forward(self, inputs):
        decoded = self.decoder(self._encode_inputs(inputs))
        if any(size % self.output_scale for size in decoded.shape[-2:]):
            raise ValueError("Decoded grid must be divisible by output_scale")
        shared = F.avg_pool2d(decoded, self.output_scale)
        features = self.interaction({task: self.adapters[task](shared) for task in self.tasks})
        return self._constrain_outputs(
            {task: self.heads[task](features[task]) for task in self.tasks})
