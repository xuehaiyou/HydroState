"""Randomly initialized U-Net for fixed five-slot retrospective retrieval."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.data.retrieval.contract import (
    INPUT_GROUPS,
    PROFILE,
    QUALITY_CHANNELS,
    SENSORS,
    TIME_STEPS,
)
from hydrostate.registry import MODELS

from .hydrostate_model import HydroStateModel
from .retrieval_model import RetrievalModelMixin


class UNetBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


@MODELS.register_module()
class UNetEncoder(nn.Module):
    """Concatenate ordered spectra, quality and masks; retain spatial skip features."""

    def __init__(self, input_profile=PROFILE, channels=(64, 128, 256, 512)):
        super().__init__()
        if input_profile != PROFILE:
            raise ValueError("U-Net requires the current five-slot input profile")
        if len(channels) != 4 or any(type(c) is not int or c < 1 for c in channels):
            raise ValueError("U-Net requires four positive integer channel widths")
        self.input_profile, self.input_keys = input_profile, tuple(SENSORS)
        self.channels, self.out_channels = tuple(channels), channels[-1]
        # 125 spectra + 120 quality statistics + 30 availability masks + 2 season channels.
        self.in_channels = TIME_STEPS * (
            sum(SENSORS.values()) + len(INPUT_GROUPS) * (QUALITY_CHANNELS + 1)
        ) + 2
        self.blocks = nn.ModuleList(
            UNetBlock(a, b) for a, b in zip((self.in_channels, *channels[:-1]), channels)
        )

    def pack_inputs(self, inputs):
        reference = inputs[self.input_keys[0]]
        batch, _, _, height, width = reference.shape
        if height % 8 or width % 8:
            raise ValueError("U-Net input grid must be divisible by eight")
        sensor_valid = inputs["sensor_valid"].to(device=reference.device, dtype=torch.bool)
        if sensor_valid.shape != (batch, TIME_STEPS, len(SENSORS)):
            raise ValueError("U-Net requires five ordered satellite slots")
        parts = []
        for index, (sensor, channels) in enumerate(SENSORS.items()):
            value = inputs[sensor]
            valid = inputs[f"{sensor}_valid"].bool()
            if value.shape != (batch, TIME_STEPS, channels, height, width) or (
                valid.shape != (batch, TIME_STEPS, 1, height, width)
            ):
                raise ValueError(f"Invalid U-Net spectral input shape: {sensor}")
            valid = valid & sensor_valid[:, :, index, None, None, None]
            # Clear excluded values before convolution, including NaN and infinity.
            parts.append(value.masked_fill(~valid, 0).flatten(1, 2))
        for name in INPUT_GROUPS:
            quality, valid = inputs[f"{name}_quality"], inputs[f"{name}_valid"].bool()
            if quality.shape != (batch, TIME_STEPS, QUALITY_CHANNELS, height, width) or (
                valid.shape != (batch, TIME_STEPS, 1, height, width)
            ):
                raise ValueError(f"Invalid U-Net observation quality shape: {name}")
            parts.extend(((quality / 8).masked_fill(~valid, 0).flatten(1, 2),
                          valid.float().flatten(1, 2)))
        season = inputs["target_season"]
        if season.shape != (batch, 2):
            raise ValueError("U-Net requires two target-season channels from the preprocessor")
        parts.append(season[:, :, None, None].expand(-1, -1, height, width))
        return torch.cat(parts, dim=1)

    def forward(self, inputs):
        value = self.pack_inputs(inputs)
        features = []
        for index, block in enumerate(self.blocks):
            if index:
                value = F.max_pool2d(value, 2)
            value = block(value)
            features.append(value)
        return tuple(features)


@MODELS.register_module()
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), channels=64):
        super().__init__()
        if len(encoder_channels) != 4 or channels != encoder_channels[0]:
            raise ValueError("U-Net decoder must match four encoder levels and the shallow width")
        self.encoder_channels = tuple(encoder_channels)
        self.out_channels = channels
        self.blocks = nn.ModuleList(
            UNetBlock(deep + skip, skip)
            for deep, skip in zip(reversed(encoder_channels[1:]), reversed(encoder_channels[:-1]))
        )

    def forward(self, features):
        value = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return value


@MODELS.register_module()
class UNetHydroStateModel(RetrievalModelMixin, HydroStateModel):
    """One shared U-Net and independent heads, with the same physical supervision."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.encoder, UNetEncoder) or not isinstance(self.decoder, UNetDecoder):
            raise TypeError("UNetHydroStateModel requires UNetEncoder and UNetDecoder")
        if self.encoder.channels != self.decoder.encoder_channels:
            raise ValueError("U-Net encoder and decoder channel widths disagree")
        if self.input_fusion is not None:
            raise ValueError("U-Net consumes observation quality directly in its input channels")

    def _forward(self, inputs):
        if "target_time" not in inputs or set(inputs["target_time"]) != set(self.tasks):
            raise ValueError("U-Net requires explicit target windows for every task")
        bounds = inputs["slot_bounds"]
        if bounds.ndim != 3 or bounds.shape[1:] != (TIME_STEPS, 2):
            raise ValueError("U-Net requires five ordered eight-day windows")
        torch._assert_async(
            ((bounds[:, :, 1] - bounds[:, :, 0] == 8 * 86400).all()
             & (bounds[:, 1:, 0] == bounds[:, :-1, 1]).all()),
            "U-Net slots must be consecutive eight-day windows",
        )
        # The final PML interval of a year may end before the full central slot.
        reference = inputs["target_time"][self.tasks[0]]
        torch._assert_async(
            ((reference[:, 0] == bounds[:, 2, 0]).all()
             & (reference[:, 1] > reference[:, 0]).all()
             & (reference[:, 1] <= bounds[:, 2, 1]).all()),
            "U-Net predicts only the central target window",
        )
        for task in self.tasks[1:]:
            torch._assert_async((inputs["target_time"][task] == reference).all(),
                                "U-Net requires aligned target windows for all tasks")
        return super()._forward(inputs)
