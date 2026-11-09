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
        embeddings = {
            name: self.task_embedding.weight[i][None, :, None, None].expand(b, -1, h, w)
            for i, name in enumerate(self.experts)
        }
        keys, values = {}, {}
        needed = {source for sources in self.sources.values() for source in sources}
        for source in self.experts:
            if source not in needed:
                continue
            feature = features[source].detach() if self.detach_source else features[source]
            feature = feature.masked_fill(~valid, 0)
            keys[source] = self.key(torch.cat((feature, embeddings[source]), dim=1))
            values[source] = self.value(feature)
        result = {}
        for target, sources in self.sources.items():
            baseline = features[target].masked_fill(~valid, 0)
            if not sources:
                result[target] = baseline
                continue
            query = self.query(torch.cat((state, embeddings[target]), dim=1))
            update = 0
            for source in sources:
                # FP32 dot products keep sigmoid scores stable under AMP.
                score = (query.float() * keys[source].float()).sum(1, keepdim=True) * self.scale
                gate = score.sigmoid().masked_fill(~valid, 0).to(values[source].dtype)
                update = update + gate * values[source]
            result[target] = (baseline + self.gamma[target] * update / len(sources)).masked_fill(
                ~valid, 0)
        return result
