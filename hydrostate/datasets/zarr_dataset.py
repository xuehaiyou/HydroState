"""Manifest-driven reader for immutable HydroState Zarr shards."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from mmengine.registry import FUNCTIONS
from torch.utils.data import Dataset, default_collate

from hydrostate.registry import DATASETS

MODALITY_KEYS = ("s1", "s2", "landsat")
TARGET_KEYS = ("water", "soil_moisture", "et")


@FUNCTIONS.register_module()
def hydro_collate(batch):
    """Pad only the varying number of complete PML cells in each window."""
    count = max(item["data_samples"]["et_native_value"].shape[0] for item in batch)
    for item in batch:
        targets = item["data_samples"]
        for key in ("et_native_weights", "et_native_value", "et_native_valid"):
            value = targets[key]
            padding = value.new_zeros((count - value.shape[0], *value.shape[1:]))
            targets[key] = torch.cat((value, padding))
    return default_collate(batch)


class _ShardCache:
    """Small per-worker LRU cache of open Zarr groups."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max_open
        self._groups: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self._groups:
            self._groups.move_to_end(path)
            return self._groups[path]
        group = zarr.open_group(path, mode="r")
        self._groups[path] = group
        if len(self._groups) > self.max_open:
            self._groups.popitem(last=False)
        return group


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def _tensor(array: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    # copy=True avoids read-only NumPy buffers and detaches data from the store.
    out = torch.from_numpy(np.array(array, copy=True))
    return out.to(dtype=dtype) if dtype is not None else out


@DATASETS.register_module()
class HydroStateZarrDataset(Dataset):
    """Read sampled multimodal windows from sharded Zarr stores.

    The manifest must contain ``sample_id``, ``split``, ``input_shard`` and
    ``input_row``. Label columns are optional for prediction, but training
    manifests should also contain ``label_shard`` and ``label_row``.
    """

    META_COLUMNS = (
        "sample_id",
        "tile_id",
        "site_id",
        "target_time",
        "label_source",
    )

    def __init__(
        self,
        manifest: str,
        split: str,
        data_root: str | None = None,
        max_open_shards: int = 8,
        training: bool = False,
        flip_probability: float = 0.0,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        table = pd.read_parquet(self.manifest_path)
        required = {"sample_id", "split", "input_shard", "input_row"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        self.table = table.loc[table["split"] == split].reset_index(drop=True)
        if self.table.empty:
            raise ValueError(f"no samples for split={split!r} in {manifest}")
        self.training = training
        self.flip_probability = flip_probability
        self._cache = _ShardCache(max_open=max_open_shards)

    def __len__(self) -> int:
        return len(self.table)

    def sampling_weights(self) -> torch.Tensor:
        """Return normalized manifest weights for ``HydroBalancedSampler``."""
        if "sample_weight" not in self.table:
            return torch.ones(len(self), dtype=torch.double)
        weights = torch.as_tensor(self.table["sample_weight"].to_numpy(), dtype=torch.double)
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("sample_weight values must be finite and positive")
        return weights

    def _load_inputs(self, row: pd.Series) -> dict[str, torch.Tensor]:
        path = _resolve_path(str(row.input_shard), self.data_root)
        group = self._cache.get(path)
        index = int(row.input_row)
        inputs = {key: _tensor(group[key][index], torch.float32) for key in MODALITY_KEYS}
        for key in ("sensor_valid", "observation_time"):
            if key not in group:
                raise KeyError(f"{path} is missing required array {key!r}")
            inputs[key] = _tensor(group[key][index])
        for key in MODALITY_KEYS:
            valid_key = f"{key}_valid"
            if valid_key in group:
                inputs[valid_key] = _tensor(group[valid_key][index], torch.bool)
        return inputs

    def _load_targets(self, row: pd.Series, height: int, width: int) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        has_labels = "label_shard" in row.index and pd.notna(row.get("label_shard"))
        if has_labels:
            path = _resolve_path(str(row.label_shard), self.data_root)
            group = self._cache.get(path)
            index = int(row.label_row)
            for key in TARGET_KEYS:
                if key in group:
                    targets[key] = _tensor(group[key][index], torch.float32)
                valid_key = f"{key}_valid"
                if valid_key in group:
                    targets[valid_key] = _tensor(group[valid_key][index], torch.bool)
            if f"et_native/{index}" in group:
                native = group[f"et_native/{index}"]
                for key in ("weights", "value", "valid"):
                    targets[f"et_native_{key}"] = _tensor(
                        native[key][:], torch.bool if key == "valid" else torch.float32
                    )
            elif "et_valid" in targets and targets["et_valid"].any():
                raise ValueError(
                    "ET labels lack native weights; re-ingest legacy samples before training"
                )
            for key in ("site_value", "site_valid", "site_footprint"):
                if key in group:
                    dtype = torch.bool if key == "site_valid" else torch.float32
                    targets[key] = _tensor(group[key][index], dtype)

        for key in TARGET_KEYS:
            targets.setdefault(key, torch.zeros((1, height, width), dtype=torch.float32))
            targets.setdefault(f"{key}_valid", torch.zeros((1, height, width), dtype=torch.bool))
        targets.setdefault("et_native_weights", torch.zeros((1, height // 3, width // 3)))
        targets.setdefault("et_native_value", torch.zeros(1))
        targets.setdefault("et_native_valid", torch.zeros(1, dtype=torch.bool))
        # Select before flipping: the center of an even-sized grid is ambiguous.
        for suffix in ("", "_valid"):
            value = targets["soil_moisture" + suffix]
            targets["soil_moisture" + suffix] = value[
                ..., height // 2 : height // 2 + 1, width // 2 : width // 2 + 1
            ]
        return targets

    def _augment(self, sample: dict[str, Any]) -> None:
        if not self.training or self.flip_probability <= 0:
            return
        dims: list[int] = []
        if torch.rand(()) < self.flip_probability:
            dims.append(-1)
        if torch.rand(()) < self.flip_probability:
            dims.append(-2)
        if not dims:
            return
        for container_name in ("inputs", "data_samples"):
            container = sample[container_name]
            for key, value in container.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim >= 2
                    and (
                        key == "et_native_weights"
                        or value.shape[-2:] == sample["inputs"]["s1"].shape[-2:]
                    )
                ):
                    container[key] = torch.flip(value, dims=dims)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        inputs = self._load_inputs(row)
        height, width = inputs["s1"].shape[-2:]
        data_samples = self._load_targets(row, height, width)
        # Strings have a stable default_collate representation; pandas timestamps,
        # missing values, and arbitrary Python objects do not.
        data_samples["metainfo"] = {
            key: "" if pd.isna(row.get(key)) else str(row.get(key)) for key in self.META_COLUMNS
        }
        sample = {"inputs": inputs, "data_samples": data_samples}
        self._augment(sample)
        return sample
