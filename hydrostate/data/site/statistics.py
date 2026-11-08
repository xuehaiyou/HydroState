"""Training-only moments for accepted station observations, with their own provenance."""

import pandas as pd

from hydrostate.data.site.contract import OBSERVATIONS, SiteSchema
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.normalization import training_fingerprint
from hydrostate.statistics import RunningMoments


def station_statistics(manifest, data_root=None):
    dataset = SiteDataset(manifest, "train", data_root=data_root)
    moments = {name: RunningMoments() for name in OBSERVATIONS}
    for _, row in dataset.table.iterrows():
        schema = SiteSchema(observation_capacity=int(row.observation_capacity))
        labels = dataset._arrays(row, "label", schema.targets)
        for name in OBSERVATIONS:
            moments[name].update(
                labels[f"{name}_native_value"][labels[f"{name}_native_valid"].astype(bool)]
            )
    observations = {name: values.result() for name, values in moments.items() if values.count}
    if not observations:
        raise ValueError("No valid station training observations")
    return dict(
        version="site_training_statistics_v1",
        split="train",
        ddof=0,
        training_fingerprint=training_fingerprint(pd.read_parquet(manifest), dataset.data_contract),
        data_contract=dataset.data_contract,
        observations=observations,
    )


def cohort_statistics(datasets):
    """Merge training moments while preserving each cohort's production contract."""
    parts = [station_statistics(item["manifest"], item["data_root"]) for item in datasets]
    observations = {}
    for name in OBSERVATIONS:
        values = [part["observations"][name] for part in parts if name in part["observations"]]
        if not values:
            continue
        count = sum(v["count"] for v in values)
        mean = sum(v["count"] * v["mean"] for v in values) / count
        variance = (
            sum(v["count"] * (v["std"] ** 2 + (v["mean"] - mean) ** 2) for v in values) / count
        )
        observations[name] = dict(count=count, mean=mean, std=variance**0.5)
    return dict(
        version="site_training_statistics_v2",
        split="train",
        ddof=0,
        observations=observations,
        cohorts=parts,
    )
