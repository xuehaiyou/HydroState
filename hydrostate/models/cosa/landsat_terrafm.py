"""Frozen SSL4EO-L and TerraFM inputs for the direct two-source BDTA decoder."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.data.retrieval.specs import INPUT_SPECS
from hydrostate.registry import MODELS

from .direct_bdta import DirectBDTADecoder
from .foundation_v2 import FoundationStageModelV2


def band_order(sensor, names):
    stored = [band.name for band in INPUT_SPECS[sensor].bands]
    return [stored.index(name) for name in names]


def align_level(feature, native_valid, support, stride, first_center):
    """Sample native features at 40 m grid centers, using 10 m pixel-edge coordinates."""
    h, w = feature.shape[-2:]
    gh, gw = support.shape[-2:]
    y = ((torch.arange(gh, device=feature.device) + 0.5) * 4 - first_center) / stride
    x = ((torch.arange(gw, device=feature.device) + 0.5) * 4 - first_center) / stride
    gy, gx = torch.meshgrid(2 * (y + 0.5) / h - 1, 2 * (x + 0.5) / w - 1, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1).expand(feature.shape[0], -1, -1, -1)
    # Exclude empty native cells from interpolation, also at partly observed edges.
    kwargs = dict(align_corners=False, padding_mode="border")
    weights = F.grid_sample(native_valid.float(), grid, **kwargs)
    values = F.grid_sample(feature.float().masked_fill(~native_valid, 0), grid, **kwargs)
    return (values / weights.clamp_min(1e-6)).masked_fill(~support, 0).to(feature.dtype)


@MODELS.register_module()
class LandsatTerraEncoder(nn.Module):
    method = "bdta"
    mixture = False
    out_channels = 768
    input_keys = ("s1", "s2", "landsat")
    landsat_widths = (64, 256, 512, 1024, 2048)

    def __init__(self, weights_root, input_profile):
        super().__init__()
        if input_profile != PROFILE:
            raise ValueError("SSL4EO-L/TerraFM requires the calibrated retrieval Zarr inputs")
        root = Path(weights_root)
        module = importlib.util.spec_from_file_location(
            "hydrostate_terrafm_release", root / "terrafm.py"
        )
        release = importlib.util.module_from_spec(module)
        module.loader.exec_module(release)
        self.terra = release.terrafm_base()
        self.terra.load_state_dict(
            torch.load(root / "TerraFM-B.pth", map_location="cpu", weights_only=True)
        )
        self.landsat = resnet50(weights=None)
        self.landsat.conv1 = nn.Conv2d(11, 64, 7, stride=2, padding=3, bias=False)
        self.landsat.fc = nn.Identity()
        self.landsat.load_state_dict(
            torch.load(
                root / "resnet50_landsat_oli_tirs_toa_moco-de7f5e0f.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
            load_computed_config,
        )

        stats = load_computed_config()
        for key, names in (("s1", release.S1_BANDS), ("s2", release.S2_BANDS)):
            names = [name.replace("B0", "B") for name in names]
            indices = band_order(key, names)
            self.register_buffer(key + "_order", torch.tensor(indices))
            spec = INPUT_SPECS[key]
            old = [stats[spec.modality][spec.bands[i].normalization_key] for i in indices]
            new = release.STATS["s2_l2a" if key == "s2" else "s1"]
            gain = torch.tensor([v["std"] for v in old]) / torch.tensor(new["std"])
            bias = (
                (torch.tensor([v["mean"] for v in old]) - torch.tensor(new["mean"]))
                / (4 * torch.tensor(new["std"]))
                + 0.5
                - 0.5 * gain
            )
            self.register_buffer(key + "_gain", gain.view(1, -1, 1, 1))
            self.register_buffer(key + "_bias", bias.view(1, -1, 1, 1))
        indices = band_order("landsat", [f"B{i}" for i in range(1, 12)])
        self.register_buffer("landsat_order", torch.tensor(indices))
        bands = [INPUT_SPECS["landsat"].bands[i] for i in indices]
        self.register_buffer("divisors", torch.tensor([b.divisor for b in bands]).view(1, -1, 1, 1))
        self.register_buffer("offsets", torch.tensor([b.offset for b in bands]).view(1, -1, 1, 1))
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def landsat_input(self, values):
        physical = (values[:, self.landsat_order].float() - 0.5) * self.divisors + self.offsets
        physical = F.avg_pool2d(physical, 3)
        # SSL4EO-L authors' uint8 compression, followed by the checkpoint's /255 transform.
        optical = physical[:, :9] / 0.4
        thermal = (physical[:, 9:] - 248.15) / 70
        return torch.cat((optical, thermal), dim=1).clamp_(0, 1).mul_(255).floor_().div_(255)

    def landsat_levels(self, values, valid30, support):
        x = self.landsat_input(values).masked_fill(~valid30, 0)
        core = self.landsat
        x = core.relu(core.bn1(core.conv1(x)))
        levels = [x]
        x = core.maxpool(x)
        for layer in (core.layer1, core.layer2, core.layer3, core.layer4):
            x = layer(x)
            levels.append(x)
        maps = []
        for feature, step in zip(levels, (2, 4, 8, 16, 32), strict=True):
            mask = F.max_pool2d(valid30.float(), step + 1, step, step // 2).bool()
            maps.append(align_level(feature, mask, support, 3 * step, 1.5))
        return maps

    def terra_levels(self, key, values, valid, support):
        x = values[:, getattr(self, key + "_order")].float()
        x = x * getattr(self, key + "_gain") + getattr(self, key + "_bias")
        pad = (0, -x.shape[-1] % 16, 0, -x.shape[-2] % 16)
        x = F.pad(x.masked_fill(~valid, 0), pad)
        patch_mask = F.max_pool2d(F.pad(valid.float(), pad), 16).bool()
        tokens = self.terra.prepare_tokens(x, is_l2a=(key == "s2"))
        levels = [tokens]
        for i, block in enumerate(self.terra.blocks, 1):
            tokens = block(tokens)
            if i in (3, 6, 9, 12):
                levels.append(tokens)
        return [
            align_level(
                t[:, 1:].transpose(1, 2).reshape(x.shape[0], -1, *patch_mask.shape[-2:]),
                patch_mask,
                support,
                16,
                8,
            )
            for t in levels
        ]

    @torch.no_grad()
    def forward(self, inputs):
        b, t = inputs["landsat"].shape[:2]
        times = inputs["slot_bounds"].sum(-1) // 2
        maps, masks = {}, {}
        for key in self.input_keys:
            x = inputs[key].flatten(0, 1)
            valid = inputs[key + "_valid"].flatten(0, 1).all(1, keepdim=True)
            support = F.max_pool2d(valid.float(), 4).bool()
            if key == "landsat":
                valid30 = F.avg_pool2d(valid.float(), 3).eq(1)
                native_support = valid30.repeat_interleave(3, -2).repeat_interleave(3, -1)
                support = F.max_pool2d(native_support.float(), 4).bool()
                maps[key] = self.landsat_levels(x, valid30, support)
            else:
                maps[key] = self.terra_levels(key, x, valid, support)
            masks[key] = support
        count = masks["s1"].int() + masks["s2"].int()
        # Both sensors describe the same slot. Available features share TerraFM's embedding space.
        terra = [(a + c) / count.clamp_min(1) for a, c in zip(maps["s1"], maps["s2"], strict=True)]
        return {
            key: [
                dict(
                    features=x.unflatten(0, (b, t)),
                    valid=mask[:, 0].unflatten(0, (b, t)),
                    time=times,
                )
                for x in levels
            ]
            for key, levels, mask in (
                ("terrafm", terra, count > 0),
                ("landsat", maps["landsat"], masks["landsat"]),
            )
        }


@MODELS.register_module()
class LandsatTerraBDTADecoder(DirectBDTADecoder):
    source_names = ("terrafm", "landsat")

    def __init__(self, **kwargs):
        super().__init__(secondary_widths=LandsatTerraEncoder.landsat_widths, **kwargs)


@MODELS.register_module()
class LandsatTerraBDTA(FoundationStageModelV2):
    def _prepare_features(self, inputs):
        levels = self.encoder(inputs)
        return {
            "terrafm": self.decoder.prepare(levels["terrafm"], inputs),
            "landsat": self.decoder.prepare(
                levels["landsat"], inputs, projections=self.decoder.anysat_projections
            ),
        }
