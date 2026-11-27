"""Uniform product/site/manual windows; only masks distinguish manual supervision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.data.site.contract import SiteSchema
from hydrostate.registry import DATASETS
from hydrostate.storage.packs import file_hash

from .joint_dataset import JointRetrievalDataset
from .manual_cover_dataset import ManualCoverDataset
from .retrieval_dataset import as_tensors


@DATASETS.register_module()
class ManualJointDataset(JointRetrievalDataset):
    def __init__(self, manual_root, manual_selection, glacier_input_root=None, specialist_sources=None, hls_root=None, **kwargs):
        super().__init__(**kwargs)
        self.training = kwargs.get("training", False)
        self.specialist_sources = specialist_sources
        self.hls_root = hls_root
        if specialist_sources:
            for component in self.components:
                component.training = False
        self.base_length = len(self)
        split = "val" if self.split == "validation" else self.split
        self.manual = ManualCoverDataset(manual_root, split, glacier_input_root=glacier_input_root)
        selection = Path(manual_selection)
        binding = json.loads(selection.with_suffix(".json").read_text())
        if binding["sha256"] != file_hash(selection):
            raise ValueError("Manual selection changed")
        if binding["publication_sha256"] != file_hash(Path(manual_root) / "_PUBLISHED.json"):
            raise ValueError("Manual selection publication changed")
        selected = pd.read_parquet(selection)
        keep = selected[selected.split.eq(split)].sample_id
        if not keep.isin(self.manual.table.sample_id).all():
            raise ValueError("Selection alters manual split")
        self.manual.table = self.manual.table[self.manual.table.sample_id.isin(keep)].reset_index(
            drop=True
        )
        if self.manual.table.empty:
            raise ValueError("Empty manual split")
        self.offsets.append(self.offsets[-1] + len(self.manual))

    def __getitem__(self, index):
        manual = index >= self.base_length
        if manual:
            item = self.manual[index - self.base_length]
            raw = item["data_samples"]
            targets = as_tensors(
                {
                    k: np.zeros(s.tail_shape, s.dtype)
                    for schema in (RetrievalSchema(), SiteSchema())
                    for k, s in schema.targets.items()
                }
            )
            for name in ("water", "snow_ice", "land"):
                targets["manual_" + name] = raw[name]
                targets["manual_" + name + "_valid"] = raw[name + "_valid"]
            row = self.manual.table.iloc[index - self.base_length]
            info = dict(
                raw["metainfo"],
                supervision_source="manual",
                spatial_group_id=row.spatial_group,
                year=pd.Timestamp(row.target_time).year,
                target_time=row.target_time,
            )
            item["data_samples"] = targets
            if self.training and not self.specialist_sources:
                for axis in (-1, -2):
                    if torch.rand(()) < 0.5:
                        for arrays in (item["inputs"], targets):
                            for k, v in arrays.items():
                                if (
                                    torch.is_tensor(v)
                                    and v.ndim >= 3
                                    and v.shape[-2:] in ((40, 40), (120, 120))
                                ):
                                    arrays[k] = v.flip(axis)
        else:
            item = super().__getitem__(index)
            targets = item["data_samples"]
            info = targets["metainfo"]
            for name in ("water", "snow_ice", "land"):
                targets["manual_" + name] = torch.zeros((1, 40, 40))
                targets["manual_" + name + "_valid"] = torch.zeros((1, 40, 40), dtype=torch.bool)
        if self.specialist_sources:
            from .specialists import specialist_inputs
            inputs = specialist_inputs(item["inputs"], info["sample_id"], self.specialist_sources,
                                       self.hls_root)
            item["inputs"] = inputs
            if self.training:
                for axis in (-1, -2):
                    if torch.rand(()) < 0.5:
                        for arrays in (inputs, targets):
                            for key, value in arrays.items():
                                if (torch.is_tensor(value) and value.ndim >= 3 and
                                        value.shape[-2:] in ((40, 40), (120, 120))):
                                    arrays[key] = value.flip(axis)
        # Identical metainfo keys are required by default_collate across source types.
        targets["metainfo"] = {
            key: info.get(key, default)
            for key, default in {
                "sample_id": "",
                "sampling_group": "-1",
                "frequency_bin": "-1",
                "target_time": "",
                "climate_class": "",
                "supervision_source": "",
                "spatial_group_id": "",
                "year": 0,
                "source": "",
                "split": self.split,
                "region": "",
            }.items()
        }
        return item
