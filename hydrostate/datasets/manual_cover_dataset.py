"""Human fraction labels with unchanged retrieval inputs and per-variable masks."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.cover_points.contract import config_identity
from hydrostate.data.manual_cover.contract import VERSION, ManualCoverSchema
from hydrostate.data.retrieval.alignment import align_inputs
from hydrostate.data.retrieval.contract import TASKS
from hydrostate.registry import DATASETS
from hydrostate.storage.packs import file_hash
from hydrostate.storage.reader import ShardCache

from .retrieval_dataset import as_tensors, read_shard_arrays


@DATASETS.register_module()
class ManualCoverDataset(Dataset):
    def __init__(self, root, split=None, *, require_published=True):
        self.root = Path(root)
        if require_published:
            pub = json.loads((self.root / "_PUBLISHED.json").read_text())
            if not pub.get("complete"):
                raise ValueError("Manual cover release is incomplete")
            for name, digest in pub["metadata_hashes"].items():
                if file_hash(self.root / name) != digest:
                    raise ValueError(f"Published metadata changed: {name}")
        self.config = json.loads((self.root / "collection.json").read_text())
        if self.config["version"] != VERSION:
            raise ValueError("Manual cover contract mismatch")
        self.config_hash = config_identity(self.config)
        self.table = pd.read_parquet(self.root / "samples.parquet")
        if (
            self.table.sample_id.duplicated().any()
            or not self.table.split.isin(["train", "val", "test"]).all()
        ):
            raise ValueError("Invalid manual sample index")
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("Unknown split")
            self.table = self.table[self.table.split.eq(split)].reset_index(drop=True)
        self.cache, self.schema = ShardCache(), ManualCoverSchema()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        raw = read_shard_arrays(
            self.root,
            self.cache,
            row,
            "input",
            self.schema.inputs,
            row.input_schema_version,
            row.input_config_sha256,
        )
        labels = read_shard_arrays(
            self.root, self.cache, row, "label", self.schema.targets, VERSION, self.config_hash
        )
        inputs = as_tensors(align_inputs(raw, self.schema))
        inputs["target_time"] = {
            task: torch.from_numpy(raw["target_bounds"].copy()) for task in TASKS
        }
        targets = as_tensors(labels)
        targets["metainfo"] = {k: str(row[k]) for k in ("sample_id", "source", "split", "region")}
        return dict(inputs=inputs, data_samples=targets)
