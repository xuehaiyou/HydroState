"""Independent station Zarr reader with the same input/augmentation contract."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.contract import OUTPUT_SCALE, PROFILE, TASKS, RetrievalSchema
from hydrostate.data.site.contract import VERSION, SiteSchema
from hydrostate.registry import DATASETS
from hydrostate.storage.reader import ShardCache
from hydrostate.tasks import task_config

from .retrieval_dataset import RetrievalDataset, as_tensors, read_shard_arrays


@DATASETS.register_module()
class SiteDataset(RetrievalDataset):
    """Product target placeholders are invalid; station observations use independent operators."""

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
        allow_empty_split=False,
    ):
        if (
            input_profile != PROFILE
            or output_scale != OUTPUT_SCALE
            or task_config(TASKS if tasks is None else tasks) != task_config(TASKS)
        ):
            raise ValueError("Station observations require the current seven-output input contract")
        self.manifest = str(manifest)
        self.root = Path(data_root or Path(manifest).parent)
        table = pd.read_parquet(manifest)
        if table.empty or not table.version.eq(VERSION).all() or table.sample_id.duplicated().any():
            raise ValueError("Invalid station manifest version or duplicate sample IDs")
        config = json.loads((self.root / "collection.json").read_text())
        if config["version"] != VERSION or config["input_profile"] != PROFILE:
            raise ValueError("Station collection contract mismatch")
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
        if self.table.empty and not allow_empty_split:
            raise ValueError(f"Empty station {split} split")
        self.input_profile, self.tasks = PROFILE, task_config(TASKS)
        self.training, self.flip_probability = training, flip_probability
        self.cache, self.schema = ShardCache(), SiteSchema()
        self.input_packs, self.label_packs = [], []

    def _arrays(self, row, kind, specs):
        return read_shard_arrays(self.root, self.cache, row, kind, specs, VERSION, self.config_hash)

    def _load_targets(self, row, metadata):
        schema = SiteSchema(observation_capacity=int(row.observation_capacity))
        targets = {k: np.zeros(s.tail_shape, s.dtype) for k, s in RetrievalSchema().targets.items()}
        observed = self._arrays(row, "label", schema.targets)
        for name, maximum in (("site_sm", 1), ("site_et", np.inf)):
            valid = observed[f"{name}_native_valid"].astype(bool)
            value, weights = (
                observed[f"{name}_native_{suffix}"] for suffix in ("value", "weights")
            )
            if (
                not np.isfinite(weights).all()
                or (weights < 0).any()
                or np.any(np.abs(weights.sum((-2, -1))[valid] - 1) > 1e-5)
                or not np.isfinite(value[valid]).all()
                or np.any((value[valid] < 0) | (value[valid] > maximum))
            ):
                raise ValueError(f"Invalid station label/operator: {name}")
        targets.update(observed)
        times = {name: torch.from_numpy(metadata["target_bounds"].copy()) for name in self.tasks}
        return as_tensors(targets), times
