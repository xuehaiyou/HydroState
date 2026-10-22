"""Strict reader for the current five-slot product-supervised dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.contract import (
    OUTPUT_SCALE,
    PROFILE,
    TASKS,
    VERSION,
    RetrievalSchema,
)
from hydrostate.registry import DATASETS
from hydrostate.storage.reader import ShardCache
from hydrostate.tasks import task_config


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
    ):
        if (
            input_profile != PROFILE
            or output_scale != OUTPUT_SCALE
            or task_config(tasks or TASKS) != task_config(TASKS)
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
        self.data_contract = dict(
            version=VERSION,
            input_profile=PROFILE,
            config_sha256=self.config_hash,
            tasks=task_config(TASKS),
        )
        self.table = table.loc[table.split == split].reset_index(drop=True)
        if self.table.empty:
            raise ValueError(f"Empty {split} split")
        self.input_profile, self.tasks = PROFILE, task_config(TASKS)
        self.training, self.flip_probability = training, flip_probability
        self.cache, self.schema = ShardCache(), RetrievalSchema()
        self.input_packs, self.label_packs = [], []

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
        result = {}
        for name, spec in specs.items():
            array = group[name]
            if array.shape[1:] != spec.tail_shape or array.dtype != np.dtype(spec.dtype):
                raise ValueError(f"Invalid schema for {path}/{name}")
            value = torch.from_numpy(array[int(row.shard_index)].copy())
            result[name] = value.bool() if name.endswith("_valid") else value
        return result

    def __getitem__(self, index):
        row = self.table.iloc[index]
        inputs = self._arrays(row, "input", self.schema.inputs)
        targets = self._arrays(row, "label", self.schema.targets)
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
        inputs["target_time"] = {name: inputs["target_bounds"].clone() for name in TASKS}
        targets["metainfo"] = {
            name: str(row[name])
            for name in ("sample_id", "frequency_bin", "target_time", "climate_class")
        }
        return dict(inputs=inputs, data_samples=targets)
