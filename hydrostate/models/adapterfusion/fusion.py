"""Masked spatial interface to the unchanged upstream attention module."""

import torch
from torch import nn

from .vendor.modeling import BertFusion


class ResidualAdapterFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = BertFusion(dict(
            query=True, key=True, value=True, value_initialized=True,
            value_before_softmax=True, residual_before=False, temperature=False,
        ), channels, attention_probs_dropout_prob=0.0)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, target, sources, source_valid, target_valid):
        """BCHW target; BNCHW sources; BNHW validity, independent of labels.

        Upstream has no attention-mask argument. Pack equal availability patterns
        and remove invalid experts BEFORE softmax, rather than zeroing their values.
        Shared-LoRA normally has one common pattern, so this is one attention call.
        """
        b, n, c, h, w = sources.shape
        query = target.permute(0, 2, 3, 1).reshape(-1, c)
        values = sources.permute(0, 3, 4, 1, 2).reshape(-1, n, c)
        valid = source_valid.permute(0, 2, 3, 1).reshape(-1, n)
        valid = valid & target_valid.reshape(-1, 1)
        codes = (valid.to(torch.int64) * (2 ** torch.arange(n, device=target.device))).sum(-1)
        delta = torch.zeros_like(query)
        # Tiny expert axis; invalid inputs (including NaN) never enter upstream.
        for code in torch.unique(codes).tolist():
            if code == 0:
                continue
            rows = codes == code
            columns = [i for i in range(n) if code & (1 << i)]
            selected = values[rows][:, columns].unsqueeze(1)
            q = query[rows].unsqueeze(1)
            residual = torch.zeros_like(q)
            update = self.attention(q, selected, selected, residual).squeeze(1)
            delta[rows] = update.to(delta.dtype)
        result = query + self.gamma * delta
        return result.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
