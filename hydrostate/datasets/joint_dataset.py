"""Joint product/site loading with immutable source contracts and group-balanced sampling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.site.archives import sha256
from hydrostate.data.site.contract import SiteSchema
from hydrostate.data.site.training_statistics import load_site_statistics
from hydrostate.normalization import training_fingerprint
from hydrostate.registry import DATASETS

from .retrieval_dataset import RetrievalDataset
from .site_dataset import SiteDataset


@DATASETS.register_module()
class JointRetrievalDataset(Dataset):
    def __init__(
        self,
        manifest,
        site_split_root,
        split,
        data_root=None,
        training=False,
        site_fraction=0.5,
        **kwargs,
    ):
        if not 0 < site_fraction < 1:
            raise ValueError("site_fraction must be between zero and one")
        if kwargs.get("label_specs") or kwargs.get("label_packs"):
            raise ValueError("Joint station training currently supports the built-in tasks only")
        self.site_fraction = site_fraction
        self.manifest, self.split = str(manifest), split
        self.site_root = Path(site_split_root)
        report = json.loads((self.site_root / "split_report.json").read_text())
        if report["version"] != "site_frequency_task_spatial_split_v3":
            raise ValueError("Joint training requires task-covering v3 station split")
        selection_path = Path(report["joint_product_selection"])
        if sha256(selection_path) != report["joint_product_selection_sha256"]:
            raise ValueError("Product overlap exclusion selection changed")
        self.product = RetrievalDataset(
            manifest, split, data_root=data_root, training=training, **kwargs
        )
        support = json.loads((selection_path.parent / "product_support.json").read_text())
        if sha256(manifest) != support["manifest_sha256"]:
            raise ValueError("Product manifest differs from spatial-isolation audit")
        selection = pd.read_parquet(selection_path)
        if selection.sample_id.duplicated().any():
            raise ValueError("Duplicate product selection")
        original = pd.read_parquet(manifest).set_index("sample_id")
        if not selection.sample_id.isin(original.index).all() or (
            original.loc[selection.sample_id, "split"].tolist() != selection.split.tolist()
        ):
            raise ValueError("Product selection changes fixed product splits")
        keep = self.product.table.sample_id.isin(selection.sample_id)
        if split == "train" and not keep.all():
            raise ValueError("Product training set changed; recompute product statistics")
        self.product.table = self.product.table[keep].reset_index(drop=True)
        if self.product.table.empty:
            raise ValueError("No products remain after overlap exclusion")
        # Product statistics stay bound to the unchanged product training records.
        self.data_contract = self.product.data_contract
        self.input_profile = self.product.input_profile
        self.input_packs = self.product.input_packs
        definitions = json.loads((self.site_root / "datasets.json").read_text())["datasets"]
        self.sites = []
        for item in definitions:
            if item["type"] != "SiteDataset":
                raise ValueError("Unexpected site dataset type")
            source = Path(item["data_root"])
            if sha256(source / "samples.parquet") != report["source_manifests"][str(source)]:
                raise ValueError("Station source manifest changed")
            self.sites.append(
                SiteDataset(item["manifest"], split, data_root=source, training=training)
            )
        inventory = pd.read_parquet(self.site_root / "inventory.parquet")
        split_path = selection_path.parent / "station_splits.parquet"
        if sha256(split_path) != report["source_splits_sha256"]:
            raise ValueError("Original spatial split proof changed")
        station_index = pd.read_parquet(split_path)
        locked = set(
            station_index.loc[
                station_index.product_overlap_count.fillna(0).gt(0)
                | station_index.split_reason.eq("previous_development_group"),
                "spatial_group_id",
            ]
        )
        if inventory.loc[inventory.spatial_group_id.isin(locked), "split"].ne("train").any():
            raise ValueError("Product/development overlap leaked into station evaluation")
        for i, site in enumerate(self.sites):
            expected = inventory[inventory.cohort.eq(i) & inventory.split.eq(split)]
            if not site.table.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                raise ValueError("Station selection disagrees with split inventory")
        if inventory.groupby("spatial_group_id").split.nunique().gt(1).any():
            raise ValueError("Station spatial group crosses splits")
        for item in report["task_coverage"]:
            group = inventory[
                inventory.frequency_bin.eq(item["frequency_bin"])
                & inventory[f"valid_{item['task']}"].gt(0)
            ]
            for name, minimum in item["minimum_groups"].items():
                if group[group.split.eq(name)].spatial_group_id.nunique() < minimum:
                    raise ValueError("Station task coverage no longer holds")
        self.components = [self.product, *self.sites]
        self.offsets = [0]
        for component in self.components:
            self.offsets.append(self.offsets[-1] + len(component))
        self.metainfo = {}

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        component = next(i for i in range(len(self.components)) if index < self.offsets[i + 1])
        local = index - self.offsets[component]
        dataset = self.components[component]
        item = dataset[local]
        targets = item["data_samples"]
        if component == 0:
            for key, spec in SiteSchema().targets.items():
                targets[key] = torch.zeros(
                    spec.tail_shape, dtype=torch.from_numpy(np.empty((), dtype=spec.dtype)).dtype
                )
        info = targets["metainfo"]
        info["supervision_source"] = "product" if component == 0 else "site"
        info["spatial_group_id"] = (
            "" if component == 0 else str(dataset.table.iloc[local].spatial_group_id)
        )
        return item

    def validate_site_statistics(self, source):
        split_report = json.loads((self.site_root / "split_report.json").read_text())
        if split_report["statistics_status"] != "complete_train_only":
            raise ValueError("Station split statistics are not complete yet")
        report, _ = load_site_statistics(source)
        if report["version"] != "site_training_statistics_v2" or len(report["cohorts"]) != len(
            self.sites
        ):
            raise ValueError("Joint training requires matching cohort statistics")
        for site, part in zip(self.sites, report["cohorts"]):
            table = pd.read_parquet(site.manifest)
            digest = training_fingerprint(table, site.data_contract)
            if (
                digest != part["training_fingerprint"]
                or site.data_contract != part["data_contract"]
            ):
                raise ValueError("Station training split/statistics mismatch")
        return report

    def sampling_weights(self):
        weights = torch.zeros(len(self), dtype=torch.double)
        groups = self.product.table.sampling_group
        for _, indices in groups.groupby(groups).groups.items():
            weights[list(indices)] = (1 - self.site_fraction) / groups.nunique() / len(indices)
        sites = pd.concat([d.table for d in self.sites], ignore_index=True)
        # Frequency first, then task and spatial group; extra dates never increase group mass.
        for _, group in sites.groupby("frequency_bin"):
            tasks = [name for name in ("sm", "et") if group[f"valid_{name}"].gt(0).any()]
            for name in tasks:
                rows = group[group[f"valid_{name}"].gt(0)]
                for _, station in rows.groupby("spatial_group_id"):
                    mass = self.site_fraction / sites.frequency_bin.nunique() / len(tasks)
                    mass /= rows.spatial_group_id.nunique() * len(station)
                    weights[torch.tensor(station.index.to_numpy()) + len(self.product)] += mass
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid joint sampling mass")
        return weights
