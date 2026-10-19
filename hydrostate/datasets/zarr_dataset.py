"""Manifest-driven reader for immutable HydroState Zarr shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from mmengine.registry import FUNCTIONS
from torch.utils.data import Dataset, default_collate

from hydrostate.data.input_products import L2_PROFILE, LEGACY_PROFILE, landsat_l2_spec
from hydrostate.data.model_inputs import input_keys, prepare_model_inputs
from hydrostate.registry import DATASETS
from hydrostate.storage.packs import VariablePack, file_hash, validate_samples
from hydrostate.tasks import resolve_tasks, task_config

MODALITY_KEYS = ("s1", "s2", "landsat")
TARGET_KEYS = ("water", "soil_moisture", "et")


@FUNCTIONS.register_module()
def hydro_collate(batch):
    """Pad the varying number of complete source pixels independently per task."""
    native = [key.removesuffix("_value") for key in batch[0]["data_samples"]
              if key.endswith("_native_value")]
    for prefix in native:
        count = max(item["data_samples"][f"{prefix}_value"].shape[0] for item in batch)
        for item in batch:
            targets = item["data_samples"]
            keys = [f"{prefix}_{suffix}" for suffix in ("weights", "value", "valid")]
            if f"{prefix}_uncertainty" in targets:
                keys.append(f"{prefix}_uncertainty")
            for key in keys:
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
            _, evicted = self._groups.popitem(last=False)
            if hasattr(evicted.store, "close"):
                evicted.store.close()
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
        input_packs: list[dict] | None = None,
        label_packs: list[dict] | None = None,
        tasks: dict | None = None,
        output_scale: int = 3,
        target_time_queries: bool = False,
        input_profile: str = LEGACY_PROFILE,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        table = pd.read_parquet(self.manifest_path)
        validate_samples(table)
        self.tasks = resolve_tasks(tasks)
        self.target_time_queries = target_time_queries
        self.input_profile = input_profile
        self.input_keys = input_keys(input_profile)
        if type(output_scale) is not int or output_scale < 1:
            raise ValueError("output_scale must be a positive integer")
        self.output_scale = output_scale
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
        self.input_packs = self._open_packs(input_packs, table, "input")
        if input_profile == L2_PROFILE:
            packs = [p for p in self.input_packs if "landsat_l2" in p.variables]
            if len(packs) != 1:
                raise ValueError("L2 profile requires exactly one landsat_l2 input pack")
            pack = packs[0]
            spec = pack.variables["landsat_l2"]
            expected = landsat_l2_spec()
            for key in ("bands", "units", "input_profile", "normalization", "temporal"):
                if spec.get(key) != expected[key]:
                    raise ValueError(f"L2 pack has incompatible {key}")
            missing = set(self.table.sample_id) - pack.locations.keys()
            if missing:
                raise ValueError(f"L2 pack has no records for {len(missing)} {split} samples; "
                                 "finish collecting the pack before training")
        self.label_packs = self._open_packs(label_packs, table, "label")
        for pack in self.label_packs:
            for name, spec in pack.variables.items():
                if name not in self.tasks:
                    raise ValueError(f"Selected label {name} has no task configuration")
                if self.tasks[name].support == "pml_native":
                    raise ValueError("Native ET extensions require the PML geometry adapter")
                if spec["grid"] != self.tasks[name].grid:
                    raise ValueError(f"Label grid disagrees with task configuration: {name}")
        self.data_contract = None
        if (input_packs or label_packs or tasks is not None or output_scale != 3
                or target_time_queries or input_profile != LEGACY_PROFILE):
            self.data_contract = dict(
                tasks=task_config(self.tasks), output_scale=output_scale,
                packs=[dict(kind=p.metadata["kind"], variables=list(p.variables),
                            metadata_sha256=file_hash(p.path / "metadata.json"),
                            **({"max_uncertainty": p.max_uncertainty} if p.max_uncertainty else {}))
                       for p in (*self.input_packs, *self.label_packs)],
            )
            if target_time_queries:
                self.data_contract["target_time_queries"] = "observation_intervals_v1"
            if input_profile != LEGACY_PROFILE:
                self.data_contract["input_profile"] = input_profile

    def _open_packs(self, configs, table, kind):
        packs, names = [], set()
        for config in configs or []:
            config = dict(config)
            path = _resolve_path(config.pop("path"), self.data_root)
            pack = VariablePack(path, table, kind=kind, include_times=self.target_time_queries,
                                **config)
            duplicates = names.intersection(pack.variables)
            if duplicates:
                raise ValueError(f"Multiple {kind} packs supply {sorted(duplicates)}")
            names.update(pack.variables)
            packs.append(pack)
        return packs

    def _pack_values(self, packs, sample_id, *, read_values=True):
        values = {}
        for pack in packs:
            for key, array in pack.read(sample_id, self._cache, read_values=read_values).items():
                dtype = (torch.bool if key.endswith("_valid") else
                         torch.long if key.endswith("_time") else torch.float32)
                values[key] = _tensor(array, dtype)
        return values

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

    def _load_inputs(self, row: pd.Series, *, read_values=True) -> dict[str, torch.Tensor]:
        path = _resolve_path(str(row.input_shard), self.data_root)
        group = self._cache.get(path)
        index = int(row.input_row)
        base_keys = [key for key in self.input_keys if key != "landsat_l2"]
        inputs = {key: (_tensor(group[key][index], torch.float32) if read_values else
                        torch.zeros(group[key].shape[1:], dtype=torch.float32))
                  for key in base_keys}
        for key in ("sensor_valid", "observation_time"):
            if key not in group:
                raise KeyError(f"{path} is missing required array {key!r}")
            inputs[key] = _tensor(group[key][index])
        for key in base_keys:
            valid_key = f"{key}_valid"
            if valid_key in group:
                inputs[valid_key] = _tensor(group[valid_key][index], torch.bool)
        extras = self._pack_values(self.input_packs, row.sample_id, read_values=read_values)
        for pack in self.input_packs:
            for name in pack.variables:
                if extras[name].shape[-2:] != inputs["s1"].shape[-2:]:
                    raise ValueError(f"Input pack {name} is not on the sample grid")
        inputs.update(extras)
        return prepare_model_inputs(inputs, self.input_profile)

    def _load_targets(self, row: pd.Series, height: int, width: int) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        has_labels = "label_shard" in row.index and pd.notna(row.get("label_shard"))
        if has_labels:
            path = _resolve_path(str(row.label_shard), self.data_root)
            group = self._cache.get(path)
            index = int(row.label_row)
            for key in self.tasks:
                if key in group:
                    targets[key] = _tensor(group[key][index], torch.float32)
                valid_key = f"{key}_valid"
                if valid_key in group:
                    targets[valid_key] = _tensor(group[valid_key][index], torch.bool)
            if "et" in self.tasks and f"et_native/{index}" in group:
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

        # Explicit pack selection replaces that variable, including missing rows.
        targets.update(self._pack_values(self.label_packs, row.sample_id))
        for key, spec in self.tasks.items():
            if spec.support == "native":
                shape = (height // self.output_scale, width // self.output_scale)
                targets.setdefault(f"{key}_native_weights", torch.zeros((1, *shape)))
                targets.setdefault(f"{key}_native_value", torch.zeros(1))
                targets.setdefault(f"{key}_native_valid", torch.zeros(1, dtype=torch.bool))
                if targets[f"{key}_native_weights"].shape[-2:] != shape:
                    raise ValueError(f"Native weights for {key} differ from the model output grid")
                continue
            shape = ((1, 1, 1) if spec.grid == "scalar" else
                     (1, height // self.output_scale, width // self.output_scale)
                     if spec.grid == "output" else (1, height, width))
            targets.setdefault(key, torch.zeros(shape, dtype=torch.float32))
            targets.setdefault(f"{key}_valid", torch.zeros(shape, dtype=torch.bool))
            # Legacy SMAP selects the window center before spatial augmentation.
            if key == "soil_moisture" and spec.support == "window_mean":
                for suffix in ("", "_valid"):
                    value = targets[key + suffix]
                    if value.shape[-2:] != (1, 1):
                        targets[key + suffix] = value[
                            ..., height // 2:height // 2 + 1, width // 2:width // 2 + 1
                        ]
            if spec.support != "pml_native":
                if targets[key].shape != shape or targets[f"{key}_valid"].shape != shape:
                    raise ValueError(f"Label {key} does not match its declared grid: {shape}")
        if "et" in self.tasks and self.tasks["et"].support == "pml_native":
            targets.setdefault("et_native_weights", torch.zeros(
                (1, height // self.output_scale, width // self.output_scale)))
            targets.setdefault("et_native_value", torch.zeros(1))
            targets.setdefault("et_native_valid", torch.zeros(1, dtype=torch.bool))
        return targets

    def _query_times(self, row, targets):
        from hydrostate.data.target_times import epoch_seconds, product_time_bounds

        default = epoch_seconds(row.target_time)
        labels = row.get("labels", {})
        labels = json.loads(labels) if isinstance(labels, str) else labels
        labels = labels if isinstance(labels, dict) else {}
        times = {}
        packed = {name for pack in self.label_packs for name in pack.variables}
        for name, spec in self.tasks.items():
            bounds = targets.get(f"{name}_time")
            if bounds is None:
                # A replacement pack cannot inherit the old product's timestamp.
                bounds = (torch.zeros(2, dtype=torch.long) if name in packed else
                          _tensor(product_time_bounds(labels, name), torch.long))
            # Missing observations still have a well-defined inference query, but
            # their supervision mask remains false. Never substitute a missing value.
            if not bounds[0]:
                valid_key = f"{name}_native_valid" if spec.support in {
                    "native", "pml_native"} else f"{name}_valid"
                if targets[valid_key].any():
                    raise ValueError(
                        f"{name}: valid labels require actual observation time bounds; "
                        "declare time_support and store the label's _time array")
                bounds = torch.tensor([default, default], dtype=torch.long)
            times[name] = bounds.clone()
        return times

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
                        key.endswith("_native_weights")
                        or value.shape[-2:] == sample["inputs"]["s1"].shape[-2:]
                        or (container_name == "data_samples" and
                            getattr(self, "tasks", {}).get(
                                key.removesuffix("_valid").removesuffix("_uncertainty")) is not None
                            and self.tasks[key.removesuffix("_valid").removesuffix(
                                "_uncertainty")].support == "pixel")
                    )
                ):
                    container[key] = torch.flip(value, dims=dims)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        inputs = self._load_inputs(row)
        height, width = inputs["s1"].shape[-2:]
        data_samples = self._load_targets(row, height, width)
        if self.target_time_queries:
            inputs["target_time"] = self._query_times(row, data_samples)
        # Strings have a stable default_collate representation; pandas timestamps,
        # missing values, and arbitrary Python objects do not.
        data_samples["metainfo"] = {
            key: "" if pd.isna(row.get(key)) else str(row.get(key)) for key in self.META_COLUMNS
        }
        sample = {"inputs": inputs, "data_samples": data_samples}
        self._augment(sample)
        return sample
