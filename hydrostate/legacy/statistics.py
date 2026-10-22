"""Historical three-variable/variable-pack statistics, for archived experiments."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydrostate.data.input_products import LEGACY_PROFILE
from hydrostate.datasets.collate import hydro_collate
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import site_supervision_values, supervision_values
from hydrostate.normalization import (
    CONFIGURED_SUPPORT_VERSION,
    SUPPORT_VERSION,
    training_fingerprint,
)
from hydrostate.statistics import RunningMoments
from hydrostate.storage.packs import file_hash
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks, task_config


class _SupportDataset(HydroStateZarrDataset):
    """Statistics need input masks/times and labels, not satellite radiances."""

    def _load_inputs(self, row):
        return super()._load_inputs(row, read_values=False)


def compute_statistics(
    manifest,
    data_root,
    num_workers=4,
    batch_size=4,
    *,
    tasks=LEGACY_TASKS,
    input_packs=None,
    label_packs=None,
    output_scale=3,
    target_time_queries=False,
    input_profile=LEGACY_PROFILE,
):
    manifest = Path(manifest)
    content = manifest.read_bytes()
    table = pd.read_parquet(manifest)
    dataset = _SupportDataset(
        str(manifest),
        "train",
        str(data_root),
        training=False,
        tasks=tasks,
        input_packs=input_packs,
        label_packs=label_packs,
        output_scale=output_scale,
        target_time_queries=target_time_queries,
        input_profile=input_profile,
    )
    specs = resolve_tasks(tasks)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=hydro_collate,
    )
    preprocessor = HydroDataPreprocessor(
        tasks=tasks, output_scale=output_scale, input_profile=input_profile
    )
    moments = {task: RunningMoments() for task in specs}
    counts = {task: dict(product_count=0, site_count=0, product_samples=0) for task in specs}
    seen = 0
    for batch in loader:
        batch = preprocessor(batch, training=False)
        samples = batch["data_samples"]
        dummy = torch.zeros_like(samples["prediction_valid"], dtype=torch.float32)
        for task in specs:
            _, values, valid = supervision_values(task, dummy, samples, tasks=specs)
            moments[task].update(values[valid].numpy())
            counts[task]["product_count"] += int(valid.sum())
            counts[task]["product_samples"] += int(valid.reshape(len(dummy), -1).any(dim=1).sum())
            site = site_supervision_values(task, dummy, samples, tasks=specs)
            if site is not None:
                _, values, valid = site
                moments[task].update(values[valid].numpy())
                counts[task]["site_count"] += int(valid.sum())
        seen += len(dummy)
        if seen % 128 < len(dummy) or seen == len(dataset):
            print(f"Processed {seen}/{len(dataset)} training samples", flush=True)
    if manifest.read_bytes() != content:
        raise RuntimeError("Manifest changed while computing target statistics")
    for pack in (*dataset.input_packs, *dataset.label_packs):
        if file_hash(pack.path / "index.parquet") != pack.metadata["index_sha256"]:
            raise RuntimeError("Pack index changed while computing target statistics")
    configured = dataset.data_contract is not None
    report = dict(
        version=2 if configured else 1,
        split="train",
        ddof=0,
        support_version=CONFIGURED_SUPPORT_VERSION if configured else SUPPORT_VERSION,
        created_utc=datetime.now(UTC).isoformat(),
        source_manifest=str(manifest.resolve()),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        training_fingerprint=training_fingerprint(table, dataset.data_contract),
        training_samples=len(dataset),
        tasks={task: dict(**moments[task].result(), **counts[task]) for task in specs},
    )
    if configured:
        report.update(task_specs=task_config(specs), data_contract=dataset.data_contract)
    return report
