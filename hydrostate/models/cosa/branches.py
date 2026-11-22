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


class BackgroundResidual(nn.Module):
    """Separate a masked tile background from spatial deviations in latent space."""

    def __init__(self, groups, state_channels, channels, center, retain_scene_context=False):
        super().__init__()
        self.center = center
        self.retain_scene_context = retain_scene_context
        self.local = StateExpert(state_channels, channels)
        self.background = nn.ModuleDict({
            name: nn.Conv2d(state_channels, channels, 1) for name in groups})
        self.detail = nn.ModuleDict({
            name: nn.Conv2d(channels, channels, 1, bias=False) for name in groups})

    def forward(self, state, valid):
        count = valid.sum((-2, -1), keepdim=True).clamp_min(1)
        mean_state = state.masked_fill(~valid, 0).float().sum((-2, -1), keepdim=True) / count
        local_input = state if self.retain_scene_context else state - mean_state.to(state.dtype)
        local = self.local(local_input.masked_fill(~valid, 0), valid)
        # The task projections are linear without bias, so centering once before
        # them gives every task a zero-mean detail without repeated reductions.
        if self.center:
            mean_local = local.float().sum((-2, -1), keepdim=True) / count
            local = local - mean_local.to(local.dtype)
        return {name: (self.background[name](mean_state.to(state.dtype)) + project(local))
                .masked_fill(~valid, 0) for name, project in self.detail.items()}
