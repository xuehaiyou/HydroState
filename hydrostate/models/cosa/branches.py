"""Small spatial experts; exchange messages before spatial upsampling."""

from torch import nn
from torch.nn import functional as F


class SpatialResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value, valid):
        value = value.masked_fill(~valid, 0)
        hidden = F.gelu(self.first(value)).masked_fill(~valid, 0)
        return (value + self.second(hidden)).masked_fill(~valid, 0)


class StateExpert(nn.Module):
    def __init__(self, in_channels=256, channels=64):
        super().__init__()
        self.project = nn.Conv2d(in_channels, channels, 1)
        self.spatial = SpatialResidual(channels)

    def forward(self, state, valid):
        return self.spatial(F.gelu(self.project(state)), valid)


class SpatialReadout(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value, valid, size):
        value = F.interpolate(value.masked_fill(~valid, 0), size=size,
                              mode="bilinear", align_corners=False)
        valid = F.interpolate(valid.float(), size=size, mode="nearest").bool()
        value = value.masked_fill(~valid, 0)
        return F.gelu(self.refine(value)).masked_fill(~valid, 0)
