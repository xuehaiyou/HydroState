"""Training-only moments on exactly the current supervision supports."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.retrieval_dataset import RetrievalDataset, as_tensors
from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION, training_fingerprint
from hydrostate.statistics import RunningMoments
from hydrostate.storage.packs import file_hash
from hydrostate.tasks import task_config

from .alignment import sensor_masks
from .contract import INPUT_GROUPS, OUTPUT_SCALE


class _RetrievalSupportDataset(RetrievalDataset):
    def __getitem__(self, index):
        row = self.table.iloc[index]
        specs = {
            key: spec
            for key, spec in self.schema.inputs.items()
            if key in {f"{name}_valid" for name in INPUT_GROUPS}
        }
        masks = sensor_masks(
            self._arrays(row, "input", specs), (self.schema.height, self.schema.width)
        )
        return dict(
            inputs=as_tensors({f"{name}_valid": value for name, value in masks.items()}),
            data_samples=as_tensors(self._arrays(row, "label", self.schema.targets)),
        )


def compute_retrieval_statistics(cfg, num_workers=0, batch_size=4):
    """Read only validity and labels; use exactly the training SM coupling masks."""
    from hydrostate.models.water_soil_coupling import WaterSoilCoupling

    options = {k: v for k, v in cfg.train_dataloader.dataset.items() if k != "type"}
    dataset = _RetrievalSupportDataset(**dict(options, training=False, split="train"))
    original_hash = file_hash(dataset.manifest)
    coupling = WaterSoilCoupling(**cfg.model.water_soil_coupling)
    moments = {name: RunningMoments() for name in dataset.tasks}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=hydro_collate,
    )
    seen = 0
    for batch in loader:
        inputs, samples = batch["inputs"], batch["data_samples"]
        observed = torch.stack([value.any(dim=1) for value in inputs.values()]).any(dim=0)
        samples["prediction_valid"] = torch.nn.functional.avg_pool2d(
            observed.float(), OUTPUT_SCALE
        ).eq(1)
        dummy = torch.zeros_like(samples["water"])
        outputs = {name: dummy for name in dataset.tasks}
        support = coupling.prepare(outputs, samples, supervision=True)
        for name in dataset.tasks:
            _, target, valid = supervision_values(name, dummy, samples, support, dataset.tasks)
            moments[name].update(target[valid.expand_as(target)].numpy())
        seen += len(dummy)
        if seen % 128 < len(dummy) or seen == len(dataset):
            print(f"Processed {seen}/{len(dataset)} training samples", flush=True)
    if file_hash(dataset.manifest) != original_hash:
        raise RuntimeError("Manifest changed while computing target statistics")
    return dict(
        version=2,
        split="train",
        ddof=0,
        support_version=CONFIGURED_SUPPORT_VERSION,
        training_fingerprint=training_fingerprint(
            pd.read_parquet(dataset.manifest), dataset.data_contract
        ),
        training_samples=len(dataset),
        task_specs=task_config(dataset.tasks),
        manifest_sha256=original_hash,
        data_contract=dataset.data_contract,
        tasks={
            name: dict(**value.result(), product_count=value.count, station_count=0)
            for name, value in moments.items()
        },
    )


def main():
    import argparse
    import json

    from mmengine.config import Config

    from hydrostate.cli import configure_data_contract

    from .configuration import require_current_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Current training configuration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen statistics: {args.output}")
    cfg = Config.fromfile(args.config)
    require_current_config(cfg)
    configure_data_contract(cfg)
    report = compute_retrieval_statistics(cfg, num_workers=args.num_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
