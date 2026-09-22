"""Adapter from batched Zarr tensors to the official rslearn OlmoEarth wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hydrostate.registry import MODELS

MODALITIES = ("sentinel1", "sentinel2_l2a", "landsat")
INPUT_KEYS = ("s1", "s2", "landsat")


def _as_datetime(epoch_seconds: int) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).replace(tzinfo=None)



def _broadcast_sdpa(attention, q, k, v, n, attn_mask=None, **kwargs):
    """Equivalent to upstream SDPA without materializing a B×heads×N×N mask."""
    if attn_mask is not None:
        if attn_mask.ndim != 2:
            raise ValueError("Expected upstream B×N key validity mask")
        attn_mask = attn_mask[:, None, None, :]
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=attention.attn_drop.p
    )


@MODELS.register_module()
class OlmoEarthEncoder(nn.Module):
    """Load OlmoEarth v1.2 Base and return its pooled spatial feature map."""

    def __init__(
        self,
        model_id: str = "OLMOEARTH_V1_2_BASE",
        model_path: str | None = None,
        patch_size: int = 4,
        embedding_channels: int = 768,
        inputs_are_normalized: bool = True,
        gradient_checkpointing: bool = False,
        broadcast_attention_mask: bool = True,
    ) -> None:
        super().__init__()
        try:
            from olmoearth_pretrain_minimal import ModelID
            from rslearn.models.olmoearth_pretrain.model import OlmoEarth
        except ImportError as exc:
            raise ImportError(
                "OlmoEarthEncoder requires current official rslearn and "
                "olmoearth-pretrain packages. See README.md."
            ) from exc

        kwargs: dict[str, Any] = dict(
            patch_size=patch_size,
            embedding_size=embedding_channels,
            autocast_dtype=None,
            token_pooling=True,
            use_legacy_timestamps=False,
            normalize=not inputs_are_normalized,
        )
        if model_path is not None and Path(model_path).exists():
            kwargs["model_path"] = model_path
            self.source = str(Path(model_path).resolve())
        else:
            try:
                resolved_id = getattr(ModelID, model_id)
            except AttributeError as exc:
                raise ValueError(f"unknown OlmoEarth ModelID {model_id!r}") from exc
            kwargs["model_id"] = resolved_id
            self.source = model_id
        self.backbone = OlmoEarth(**kwargs)
        from functools import partial
        if broadcast_attention_mask:
            for module in self.backbone.model.modules():
                if hasattr(module, "sdpa") and module.fast_attn and not module.use_flash_attn:
                    module.sdpa = partial(_broadcast_sdpa, module)
        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            for block in self.backbone.model.blocks:
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        self.out_channels = embedding_channels
        self.patch_size = patch_size

    @staticmethod
    def _timestamps_for_sample(
        observation_time: torch.Tensor, valid: torch.Tensor, modality_index: int
    ) -> list[tuple[datetime, datetime]]:
        values = observation_time[:, modality_index]
        times: list[tuple[datetime, datetime]] = []
        for value, is_valid in zip(values.tolist(), valid.tolist()):
            if not is_valid:
                continue
            timestamp = _as_datetime(int(value))
            times.append((timestamp, timestamp))
        return times

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        from rslearn.train.model_context import ModelContext, RasterImage

        sensor_valid = inputs["sensor_valid"].bool()  # B,T,3
        observation_time = inputs["observation_time"].long()  # B,T,3 epoch seconds
        batch_size = sensor_valid.shape[0]
        context_inputs: list[dict[str, RasterImage]] = []

        for batch_index in range(batch_size):
            sample: dict[str, RasterImage] = {}
            for modality_index, (input_key, modality) in enumerate(zip(INPUT_KEYS, MODALITIES)):
                valid = sensor_valid[batch_index, :, modality_index]
                if not torch.any(valid):
                    continue
                image = inputs[input_key][batch_index, valid]  # Tv,C,H,W
                valid_key = f"{input_key}_valid"
                if valid_key in inputs:
                    pixel_valid = inputs[valid_key][batch_index, valid]
                    if pixel_valid.ndim == 3:
                        pixel_valid = pixel_valid.unsqueeze(1)
                    image = image * pixel_valid.to(dtype=image.dtype)
                image = image.permute(1, 0, 2, 3).contiguous()  # C,Tv,H,W
                timestamps = self._timestamps_for_sample(
                    observation_time[batch_index], valid, modality_index
                )
                sample[modality] = RasterImage(image=image, timestamps=timestamps)
            if not sample:
                raise ValueError(f"sample {batch_index} has no valid satellite modality")
            context_inputs.append(sample)

        context = ModelContext(inputs=context_inputs, metadatas=[])
        output = self.backbone(context)
        features = output.feature_maps[0]
        if features.shape[1] != self.out_channels:
            raise RuntimeError(
                f"expected {self.out_channels} encoder channels, got {features.shape[1]}"
            )
        return features

