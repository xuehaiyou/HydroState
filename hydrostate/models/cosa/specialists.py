"""Frozen sensor specialists with four co-registered exits and the existing TerraWeave head."""

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.registry import MODELS

from .foundation_v2 import FoundationStageModelV2


@MODELS.register_module()
class SpecialistHydroDataPreprocessor(HydroDataPreprocessor):
    def __init__(self, sources, **kwargs):
        super().__init__(**kwargs)
        self.input_keys = tuple(sources)

    def forward(self, data, training=False):
        # Specialist backbones consume slot bounds directly, including entirely empty windows.
        inputs = dict(data["inputs"])
        inputs.pop("sensor_valid")
        inputs.pop("observation_time")
        return super().forward(dict(data, inputs=inputs), training)


@MODELS.register_module()
class SpecialistEncoder(nn.Module):
    method = "terraweave"
    mixture = False
    widths = {"s2": 768, "s1": 384}
    exits = (2, 5, 8, 11)

    def __init__(self, sources, weights_root, input_profile):
        super().__init__()
        self.input_keys = tuple(sources)
        self.out_channels = self.widths[sources[0]]
        self.backbone = nn.ModuleDict()
        root = Path(weights_root)
        if "s2" in sources:
            sys.path.insert(0, str(root))
            from prithvi_mae import PrithviViT

            cfg = json.loads((root / "config.json").read_text())["pretrained_cfg"]
            core = PrithviViT(**cfg)
            state = torch.load(root / "prithvi.pt", map_location="cpu", weights_only=True)
            core.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")})
            # Encode each composite independently. The retrieval pooler attends over the five slots.
            pos = core.interpolate_pos_encoding((1, 128, 128))
            core.pos_embed = pos
            core.patch_embed.grid_size = [1, 8, 8]
            core.num_frames = 1
            self.backbone["s2"] = core
            for key in ("mean", "std"):
                self.register_buffer("prithvi_" + key, torch.tensor(cfg[key]).view(1, 6, 1, 1))
        if "s1" in sources:
            sys.path.insert(0, str(root / "softcon"))
            from models.dinov2.vision_transformer import vit_small

            core = vit_small(
                img_size=224,
                patch_size=14,
                in_chans=2,
                block_chunks=0,
                init_values=1e-5,
                num_register_tokens=0,
            )
            core.load_state_dict(
                torch.load(root / "softcon-s.pth", map_location="cpu", weights_only=True)
            )
            self.backbone["s1"] = core
        self.backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, inputs):
        outputs = {}
        times = inputs["slot_bounds"].sum(-1) // 2
        for key, core in self.backbone.items():
            x = inputs[key]
            b, t = x.shape[:2]
            x = x.flatten(0, 1)
            valid = inputs[key + "_valid"].flatten(0, 1).all(1, keepdim=True)
            if key == "s2":
                x = (x - self.prithvi_mean) / self.prithvi_std
                x = F.interpolate(x.masked_fill(~valid, 0), (128, 128), mode="nearest")
                coords = inputs["location"].repeat_interleave(t, dim=0)
                layers = core.forward_features(
                    x.unsqueeze(2), inputs["calendar"].float().reshape(-1, 1, 2), coords
                )
                levels = [
                    layers[i][:, 1:].transpose(1, 2).reshape(b * t, -1, 8, 8) for i in self.exits
                ]
            else:
                x = F.interpolate(x.clamp(0, 1).masked_fill(~valid, 0), (112, 112), mode="nearest")
                levels = core.get_intermediate_layers(x, n=self.exits, reshape=True)
            patch_valid = F.adaptive_max_pool2d(valid.float(), 8)
            support = F.adaptive_max_pool2d(valid.float(), 30).bool().reshape(b, t, 30, 30)
            outputs[key] = []
            for level in levels:
                feat = F.interpolate(
                    level * patch_valid, (30, 30), mode="bilinear", align_corners=False
                )
                weight = F.interpolate(patch_valid, (30, 30), mode="bilinear", align_corners=False)
                feat = feat / weight.clamp_min(1e-6)
                outputs[key].append(
                    dict(features=feat.reshape(b, t, -1, 30, 30), valid=support, time=times)
                )
        return outputs


@MODELS.register_module()
class SpecialistTerraWeave(FoundationStageModelV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extra_projections = nn.ModuleDict()
        for key in self.encoder.input_keys[1:]:
            width = self.encoder.widths[key]
            self.extra_projections[key] = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.decoder.out_channels))
                for _ in self.encoder.exits
            )

    def _prepare_features(self, inputs):
        return {
            key: self.decoder.prepare(
                levels,
                inputs,
                projections=(
                    self.extra_projections[key] if key in self.extra_projections else None
                ),
            )
            for key, levels in self.encoder(inputs).items()
        }
