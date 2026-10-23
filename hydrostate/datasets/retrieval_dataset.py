"""Strict reader for the current five-slot product-supervised dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.alignment import align_inputs, grid_metadata
from hydrostate.data.retrieval.contract import (
    OUTPUT_SCALE,
    PROFILE,
    TASKS,
    VERSION,
    RetrievalSchema,
)
from hydrostate.data.retrieval.label_packs import LabelPackSet
from hydrostate.data.retrieval.specs import resolve_label_specs, specification_snapshot
from hydrostate.registry import DATASETS
from hydrostate.storage.reader import ShardCache
from hydrostate.tasks import task_config


def as_tensors(arrays):
    return {
        name: torch.from_numpy(value.astype(bool) if name.endswith("_valid") else value)
        for name, value in arrays.items()
    }


@DATASETS.register_module()
class RetrievalDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        data_root=None,
        training=False,
        tasks=None,
        input_profile=PROFILE,
        output_scale=OUTPUT_SCALE,
        flip_probability=0.5,
        label_specs=None,
        label_packs=None,
    ):
        definitions = resolve_label_specs(label_specs)
        expected_tasks = {name: spec.task() for name, spec in definitions.items()}
        if (
            input_profile != PROFILE
            or output_scale != OUTPUT_SCALE
            or task_config(expected_tasks if tasks is None else tasks)
            != task_config(expected_tasks)
        ):
            raise ValueError("RetrievalDataset requires the current task/input/grid contract")
        self.manifest = str(manifest)
        self.root = Path(data_root or Path(manifest).parent)
        table = pd.read_parquet(manifest)
        if (
            table.empty
            or not (table.version == VERSION).all()
            or table.sample_id.duplicated().any()
        ):
            raise ValueError("Invalid retrieval manifest version or duplicate sample IDs")
        config = json.loads((self.root / "collection.json").read_text())
        if config["version"] != VERSION or config["input_profile"] != PROFILE:
            raise ValueError("Dataset collection contract mismatch")
        self.config_hash = fingerprint(
            {k: v for k, v in config.items() if k not in ("version", "input_profile")}
        )
        self.pack_set = LabelPackSet(label_packs, table, self.root, definitions)
        snapshot = specification_snapshot(definitions)
        self.data_contract = dict(
            version=VERSION,
            input_profile=PROFILE,
            config_sha256=self.config_hash,
            tasks=task_config(expected_tasks),
            specifications=snapshot,
            label_specs={
                name: value for name, value in snapshot["labels"].items() if name not in TASKS
            },
            label_packs=self.pack_set.identity,
        )
        self.table = table.loc[table.split == split].reset_index(drop=True)
        if self.table.empty:
            raise ValueError(f"Empty {split} split")
        self.input_profile, self.tasks = PROFILE, task_config(expected_tasks)
        self.training, self.flip_probability = training, flip_probability
        self.cache, self.schema = ShardCache(), RetrievalSchema()
        self.input_packs, self.label_packs = [], self.pack_set.packs

    def __len__(self):
        return len(self.table)

    def _arrays(self, row, kind, specs):
        path = self.root / row[f"{kind}_path"]
        if not (path / "_SUCCESS.json").exists():
            raise ValueError(f"Incomplete shard: {path}")
        group = self.cache.get(str(path))
        if (
            group.attrs.get("schema_version") != VERSION
            or group.attrs.get("config_sha256") != self.config_hash
        ):
            raise ValueError(f"Shard contract mismatch: {path}")
        if kind == "input" and group.attrs.get("input_grids") != grid_metadata(
            dict(crs=row["crs"], transform=json.loads(row["transform"]))
        ):
            raise ValueError(f"Native input grids disagree with tile bounds: {path}")
        result = {}
        for name, spec in specs.items():
            array = group[name]
            if array.shape[1:] != spec.tail_shape or array.dtype != np.dtype(spec.dtype):
                raise ValueError(f"Invalid schema for {path}/{name}")
            result[name] = array[int(row.shard_index)].copy()
        return result

    def _load_targets(self, row, metadata):
        targets = self._arrays(row, "label", self.schema.targets)
        bounds = metadata["target_bounds"]
        times = {name: bounds.copy() for name in self.tasks}
        extra, observed_times = self.pack_set.read(
            row.sample_id, self.cache, bounds, metadata["slot_bounds"][[0, -1], [0, 1]]
        )
        targets.update(extra)
        times.update(observed_times)
        return as_tensors(targets), {name: torch.from_numpy(t) for name, t in times.items()}

    def __getitem__(self, index):
        row = self.table.iloc[index]
        raw = self._arrays(row, "input", self.schema.inputs)
        targets, times = self._load_targets(row, raw)
        inputs = as_tensors(align_inputs(raw, self.schema))
        if self.training:
            for axis in (-1, -2):
                if torch.rand(()) < self.flip_probability:
                    for arrays in (inputs, targets):
                        for name, value in arrays.items():
                            if value.ndim >= 3 and value.shape[-2:] in (
                                (self.schema.height, self.schema.width),
                                (
                                    self.schema.height // OUTPUT_SCALE,
                                    self.schema.width // OUTPUT_SCALE,
                                ),
                            ):
                                arrays[name] = value.flip(axis)
        inputs["target_time"] = times
        targets["metainfo"] = {
            name: str(row[name])
            for name in (
                "sample_id",
                "sampling_group",
                "frequency_bin",
                "target_time",
                "climate_class",
            )
        }
        return dict(inputs=inputs, data_samples=targets)
