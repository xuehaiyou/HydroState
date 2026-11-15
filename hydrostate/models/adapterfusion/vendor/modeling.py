# ruff: noqa -- preserve pinned upstream code verbatim
"""AdapterHub BertFusion, extracted unchanged from pinned upstream source.

Source: adapter-hub/adapters, commit 53a1ea164c07498be7d522aed96c9e600aa3b38b
File: src/adapters/methods/modeling.py
Only dependency scaffolding is reduced: config annotation aliases dict and
Adapter retains only the upstream initialization helper. See LICENSE.
"""

import torch
from torch import nn

AdapterFusionConfig = dict


class Adapter:
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertFusion(nn.Module):
    """
    Implementation of an AdapterFusion block.
    """

    def __init__(
        self,
        config: AdapterFusionConfig,
        dense_size,
        attention_probs_dropout_prob,
    ):
        super(BertFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if not self.config["query"] and not self.config["key"] and not self.config["value"]:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config["query"]:
            self.query = nn.Linear(self.dense_size, self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config["key"]:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config["value"]:
            self.value = nn.Linear(self.dense_size, self.dense_size, bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config["value_initialized"]:
                init_tensor = (
                    torch.zeros(
                        self.dense_size,
                        self.dense_size,
                        device=self.value.weight.device,
                        dtype=self.value.weight.dtype,
                    )
                    + 0.000001
                )
                init_tensor.fill_diagonal_(1.0)
                self.value.weight.data = init_tensor
        if self.config["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual, output_attentions: bool = False):
        if self.config["residual_before"]:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config["query"]:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config["key"]:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config["value"] and self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        if self.config["value"] and not self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config["residual_before"]:
            context_layer += residual

        if output_attentions:
            attention_probs = attention_probs.detach().cpu().numpy()
            return context_layer, attention_probs
        else:
            return context_layer
