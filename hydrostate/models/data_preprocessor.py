"""MMEngine data preprocessor for already aligned Zarr samples."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.model import BaseDataPreprocessor

from hydrostate.registry import MODELS


@MODELS.register_module()
class HydroDataPreprocessor(BaseDataPreprocessor):
    """Move nested batches to the target device and enforce model dtypes.

    Satellite normalization is performed while creating Zarr samples. This is
    intentional: the training hot path should not repeat preprocessing, and the
    same preprocessing version is recorded in every shard's attributes.
    """

    def forward(self, data: dict[str, Any], training: bool = False) -> dict[str, Any]:
        data = self.cast_data(data)
        for key in ("s1", "s2", "landsat"):
            data["inputs"][key] = data["inputs"][key].float()
        data["inputs"]["sensor_valid"] = data["inputs"]["sensor_valid"].bool()
        for key, value in data["data_samples"].items():
            if key.endswith("_valid") and torch.is_tensor(value):
                data["data_samples"][key] = value.bool()
        return data

