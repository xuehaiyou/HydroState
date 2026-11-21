"""Human point labels accompany input-only inference, not product supervision."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.cover_points.contract import VERSION, CoverPointSchema, config_identity
from hydrostate.data.retrieval.alignment import align_inputs
from hydrostate.data.retrieval.contract import PROFILE, TASKS
from hydrostate.storage.packs import file_hash
from hydrostate.storage.reader import ShardCache

from .retrieval_dataset import as_tensors, read_shard_arrays


class CoverPointDataset(Dataset):
    def __init__(self, root, *, require_published=True):
        self.root = Path(root)
        if require_published and not (self.root / "_PUBLISHED.json").exists():
            raise ValueError("External cover dataset has not been published")
        if require_published:
            publication = json.loads((self.root / "_PUBLISHED.json").read_text())
            for name, key in [
                ("samples.parquet", "samples_sha256"),
                ("points.parquet", "points_sha256"),
                ("collection.json", "collection_sha256"),
            ]:
                if file_hash(self.root / name) != publication[key]:
                    raise ValueError("Published cover dataset metadata changed")
        config = json.loads((self.root / "collection.json").read_text())
        if config["version"] != VERSION or config["input_profile"] != PROFILE:
            raise ValueError("Cover point input/label contract mismatch")
        self.config_hash = config_identity(config)
        self.table = pd.read_parquet(self.root / "samples.parquet")
        if (
            self.table.empty
            or self.table.sample_id.duplicated().any()
            or not self.table.version.eq(VERSION).all()
            or not self.table.split.eq("test").all()
        ):
            raise ValueError("Invalid external-test manifest")
        self.cache = ShardCache()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        schema = CoverPointSchema(observation_capacity=int(row.observation_capacity))
        raw = read_shard_arrays(
            self.root, self.cache, row, "input", schema.inputs, VERSION, self.config_hash
        )
        points = read_shard_arrays(
            self.root, self.cache, row, "label", schema.targets, VERSION, self.config_hash
        )
        inputs = as_tensors(align_inputs(raw, schema))
        inputs["target_time"] = {
            task: torch.from_numpy(raw["target_bounds"].copy()) for task in TASKS
        }
        return dict(inputs=inputs, points=points, sample_id=str(row.sample_id))
