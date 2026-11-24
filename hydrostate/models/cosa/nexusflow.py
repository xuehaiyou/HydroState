"""NexusFlow three-task center matching with the NYUv2 MLP surrogate."""

import torch
from torch import nn
from torch.nn import functional as F


class AffineCoupling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width // 2, 512), nn.ReLU(), nn.Linear(512, width))

    def forward(self, x, inverse=False):
        a, b = x.chunk(2, -1)
        scale, shift = self.net(a).chunk(2, -1)
        scale = scale.tanh()
        b = (b - shift) * (-scale).exp() if inverse else b * scale.exp() + shift
        return torch.cat((a, b), dim=-1)


class TaskSurrogate(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.aggregate = nn.Sequential(
            nn.Linear(4 * channels, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.couplings = nn.ModuleList(AffineCoupling(width) for _ in range(4))

    def forward(self, state, valid):
        pooled = F.adaptive_avg_pool2d(state.masked_fill(~valid, 0).float(), 2)
        coverage = F.adaptive_avg_pool2d(valid.float(), 2)
        x = self.aggregate((pooled / coverage.clamp_min(1e-8)).flatten(1))
        for coupling in self.couplings:
            x = coupling(x).roll(x.shape[-1] // 2, dims=-1)
        return x


class NexusFlow(nn.Module):
    def __init__(self, channels, groups, width=256):
        super().__init__()
        self.surrogates = nn.ModuleDict({g: TaskSurrogate(channels, width) for g in groups})

    def forward(self, states, valid):
        latent = torch.stack([net(states[g], valid).float() for g, net in self.surrogates.items()])
        distance = (latent - latent.mean(0)).norm(dim=-1).sum(0)
        available = valid.flatten(1).any(-1)
        return (distance * available).sum() / available.sum().clamp_min(1)
