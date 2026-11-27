"""Align OlmoEarth and AnySat exits with explicit observation support."""

import torch
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .foundation_v2 import FoundationStageModelV2
from .terraweave import TerraWeaveDecoder


def source_weights(query, keys, available, scale):
    """Source-first tensors with feature channels last. Empty support contributes zero."""
    if scale is None:
        weights = available.float()
        return weights / weights.sum(0, keepdim=True).clamp_min(1)
    scores = F.cosine_similarity(keys.float(), query.float().unsqueeze(0), dim=-1) * scale
    scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
    return scores.softmax(0).masked_fill(~available, 0)


class MultiFMStageModel(FoundationStageModelV2):
    def __init__(self, anysat, sources=("olmoearth", "anysat"), **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.decoder, self.decoder_class):
            raise ValueError(f"{type(self).__name__} requires {self.decoder_class.__name__}")
        self.sources = tuple(sources)
        if self.sources not in (("olmoearth",), ("anysat",), ("olmoearth", "anysat")):
            raise ValueError("Unexpected active foundation models")
        if anysat is None and "anysat" in self.sources:
            raise ValueError("Active AnySat source requires an encoder configuration")
        self.anysat = MODELS.build(anysat) if anysat is not None else None
        if self.anysat is not None and not self.anysat.ladder_features:
            raise ValueError("AnySat must emit four stage exits")

    def _prepare_features(self, inputs):
        prepared = {}
        if "olmoearth" in self.sources:
            prepared.update(super()._prepare_features(inputs))
        if "anysat" in self.sources:
            prepared["anysat"] = self.decoder.prepare(
                self.anysat(inputs), inputs, projections=self.decoder.anysat_projections
            )
        return prepared


@MODELS.register_module()
class TerraWeave(MultiFMStageModel):
    decoder_class = TerraWeaveDecoder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.encoder.method != "terraweave" or (
            self.anysat is not None and (not self.anysat.frozen or self.anysat.include_patch)
        ):
            raise ValueError("TerraWeave requires four exits from each frozen encoder")
