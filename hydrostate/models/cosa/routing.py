"""Shared Q/K/V routing over an explicit directed expert graph."""

import math
import re

import torch
from torch import nn


class StateRouter(nn.Module):
    def __init__(self, experts, state_channels=256, channels=64, enabled=True,
                 task_embedding_dim=32, routing_dim=64, residual_init=0.0,
                 detach_source=True, sources=None):
        super().__init__()
        self.experts = tuple(experts)
        if (not self.experts or len(set(self.experts)) != len(self.experts)
                or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in self.experts)):
            raise ValueError("Expert names must be unique lowercase module identifiers")
        if min(state_channels, channels, task_embedding_dim, routing_dim) < 1:
            raise ValueError("Router dimensions must be positive")
        if not math.isfinite(residual_init):
            raise ValueError("Router residual initialization must be finite")
        self.enabled, self.detach_source = enabled, detach_source
        self.sources = {target: tuple(source for source in self.experts if source != target)
                        for target in self.experts} if sources is None else {
                            target: tuple(values) for target, values in sources.items()}
        if set(self.sources) != set(self.experts) or any(
            len(set(values)) != len(values) or target in values
            or not set(values).issubset(self.experts)
            for target, values in self.sources.items()
        ):
            raise ValueError("Router sources must give unique non-self experts for every target")
        # Always construct the same modules for B/C. Disabling only changes trainability
        # and forward computation, so the switch cannot shift the random initialization.
        self.task_embedding = nn.Embedding(len(self.experts), task_embedding_dim)
        self.query = nn.Conv2d(state_channels + task_embedding_dim, routing_dim, 1)
        self.key = nn.Conv2d(channels + task_embedding_dim, routing_dim, 1)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(residual_init))) for name in self.experts})
        self.scale = routing_dim ** -0.5
        if not enabled:
            self.requires_grad_(False)

    def forward(self, state, features, valid):
        if set(features) != set(self.experts):
            raise ValueError("Router features disagree with configured experts")
        if not self.enabled:
            return features
        b, _, h, w = state.shape
        count = len(self.experts)
        embeddings = self.task_embedding.weight[:, None, :, None, None].expand(
            -1, b, -1, h, w)
        stacked = torch.stack([features[name] for name in self.experts])
        source = stacked.detach() if self.detach_source else stacked
        source = source.masked_fill(~valid[None], 0)
        keys = self.key(torch.cat((source, embeddings), dim=2).flatten(0, 1)).reshape(
            count, b, -1, h, w)
        values = self.value(source.flatten(0, 1)).reshape(count, b, -1, h, w)
        queries = self.query(torch.cat((state[None].expand(count, -1, -1, -1, -1),
                                        embeddings), dim=2).flatten(0, 1)).reshape(
                                            count, b, -1, h, w)
        # Target/source expert axes are small; no spatial all-pairs tensor.
        scores = torch.einsum("ibdhw,jbdhw->ijbhw", queries.float(), keys.float()) * self.scale
        edges = scores.new_tensor([[source in self.sources[target] for source in self.experts]
                                   for target in self.experts], dtype=torch.bool)
        gates = scores.sigmoid().masked_fill(~edges[:, :, None, None, None], 0)
        gates = gates.masked_fill(~valid[:, 0][None, None], 0).to(values.dtype)
        updates = torch.einsum("ijbhw,jbchw->ibchw", gates, values)
        return {name: (features[name] + self.gamma[name] * updates[i] / len(self.sources[name])
                       ).masked_fill(~valid, 0) if self.sources[name] else
                       features[name].masked_fill(~valid, 0)
                for i, name in enumerate(self.experts)}
