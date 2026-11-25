"""AnySat pretrained patch sequences for the three-depth B temporal decoder."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hydrostate.registry import MODELS

from .encoder import CoSASharedLoRAEncoder


@MODELS.register_module()
class AnySatEncoder(nn.Module):
    input_keys = CoSASharedLoRAEncoder.input_keys
    out_channels = 768

    def __init__(self, source, weights, input_profile, ladder_features=False):
        super().__init__()
        self.input_profile = input_profile
        self.ladder_features = ladder_features
        source = Path(source).resolve()
        sys.path.insert(0, str(source))
        from hubconf import AnySat

        model = AnySat(flash_attn=False).model
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        self.backbone = model.requires_grad_(False)
        self.backbone.blocks[-1].requires_grad_(True)
        # Preserve the eleven calibrated Landsat inputs. For S2 initialize
        # the pretrained ten bands in published order; B1/B9 remain available.
        self.align = nn.ModuleDict({key: nn.Conv2d(n, m, 1) for key, n, m in
                                    (("s2", 12, 10), ("s1", 2, 3), ("landsat", 11, 11))})
        with torch.no_grad():
            for key, indices in (("s2", (0, 1, 2, 4, 5, 6, 3, 7, 8, 9)),
                                 ("landsat", range(11)), ("s1", (0, 1))):
                layer = self.align[key]
                layer.weight.zero_()
                layer.bias.zero_()
                for out, index in enumerate(indices):
                    layer.weight[out, index] = 1
            # Initial radar contrast in normalized log-power coordinates.
            self.align["s1"].weight[2, :, 0, 0] = torch.tensor([1., -1.])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _slot(self, sensors, dates, *arrays):
        data, masks = {}, []
        for key, stamp, (x, valid) in zip(sensors, dates, zip(arrays[::2], arrays[1::2])):
            modality = {"s1": "s1", "s2": "s2", "landsat": "l8"}[key]
            centered = (x.float() - 0.5) * (2 if key == "landsat" else 4)
            aligned = self.align[key](centered.masked_fill(~valid, 0))
            available = valid.all(1, keepdim=True)
            data[modality] = aligned[:, None]
            data[modality + "_mask"] = available[:, None]
            data[modality + "_dates"] = x.new_tensor([[stamp]], dtype=torch.long)
            masks.append(F.max_pool2d(available.float(), 4).bool())
        mask = torch.cat(masks, dim=1)
        h, w = mask.shape[-2:]
        intermediate = []

        def collect_tokens(tokens):
            tokens = tokens[:, 1:].reshape(1, len(sensors), h, w, -1)
            pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[..., None]
            intermediate.append(pooled.permute(0, 3, 1, 2))

        indices = (1, 3, 5) if self.ladder_features else (1, 3)
        handles = [self.backbone.blocks[i].register_forward_hook(
            lambda module, args, output: collect_tokens(output)) for i in indices]
        if self.ladder_features:
            # Initial patch state before the six main blocks and final CrossBlockMulti.
            handles.append(self.backbone.blocks[0].register_forward_pre_hook(
                lambda module, args: collect_tokens(args[0])))
        try:
            final = self.backbone.forward_release(data, scale=4, output="patch")
        finally:
            for handle in handles:
                handle.remove()
        return (*intermediate, final.permute(0, 3, 1, 2))

    def forward(self, inputs):
        b, steps, _, height, width = inputs["s1"].shape
        times = inputs["observation_time"].detach().cpu().tolist()
        present = inputs["sensor_valid"].detach().cpu().tolist()
        levels = [[] for _ in range(5 if self.ladder_features else 3)]
        masks, stamps = [], []
        for i in range(b):
            for t in range(steps):
                keys, dates, arrays, pixels = [], [], [], []
                slot_times = set()
                for sensor, key in enumerate(self.input_keys):
                    if not present[i][t][sensor]:
                        continue
                    x = inputs[key][i:i + 1, t]
                    valid = inputs[key + "_valid"][i:i + 1, t].reshape(1, -1, height, width)
                    keys.append(key)
                    stamp = times[i][t][sensor]
                    dates.append(datetime.fromtimestamp(stamp, UTC).timetuple().tm_yday - 1)
                    slot_times.add(stamp)
                    arrays.extend((x, valid))
                    pixels.append(valid.all(1, keepdim=True))
                if len(slot_times) > 1:
                    raise ValueError("AnySat joint composites require aligned slot timestamps")
                stamps.append(next(iter(slot_times), 0))
                if keys:
                    values = checkpoint(self._slot, keys, dates, *arrays, use_reentrant=False)
                    valid = F.max_pool2d(torch.stack(pixels).any(0).float(), 4).bool()
                else:
                    empty = inputs["s1"].new_zeros((1, self.out_channels, height // 4, width // 4))
                    values, valid = (empty,) * len(levels), empty[:, :1].bool()
                masks.append(valid)
                for output, value in zip(levels, values):
                    output.append(value.masked_fill(~valid, 0))
        valid = torch.cat(masks).reshape(b, steps, height // 4, width // 4)
        time = inputs["observation_time"].new_tensor(stamps).reshape(b, steps)
        return tuple(dict(features=torch.stack(values).reshape(b, steps, self.out_channels,
                                                               height // 4, width // 4),
                          valid=valid, time=time) for values in levels)
