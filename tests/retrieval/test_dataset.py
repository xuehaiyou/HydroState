import json

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine.config import Config

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.alignment import grid_metadata
from hydrostate.data.retrieval.contract import (
    INPUT_GROUPS,
    PROFILE,
    VERSION,
    RetrievalSchema,
    intervals,
)
from hydrostate.data.retrieval.sampling import SAMPLING_VERSION
from hydrostate.data.retrieval.statistics import compute_retrieval_statistics
from hydrostate.data.retrieval.validation import validate_arrays
from hydrostate.datasets.retrieval_dataset import RetrievalDataset
from hydrostate.storage import SampleShardWriter


def make_dataset(root):
    schema = RetrievalSchema()
    cfg = dict(samples=2, output_dir=str(root), sampling_version=SAMPLING_VERSION)
    (root / "collection.json").write_text(
        json.dumps(dict(cfg, version=VERSION, input_profile=PROFILE))
    )
    rows = []
    sample = dict(crs="EPSG:32650", transform=[10, 0, 400005, 0, -10, 3000015])
    for index in range(2):
        inputs = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.inputs.items()}
        labels = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.targets.items()}
        for sensor in INPUT_GROUPS:
            inputs[f"{sensor}_valid"][:] = 1
            inputs[f"{sensor}_quality"][:, 0] = 2
            inputs[f"{sensor}_quality"][:, 1:] = 1
            inputs[sensor][:] = 0.5
        # A 15 m spatial pattern detects noninteger resampling and flip mistakes.
        inputs["landsat_15m"][:, 0] = np.arange(80)[None, :]
        inputs["sensor_valid"][:] = 1
        inputs["slot_bounds"][:] = [
            [int(pd.Timestamp(v, tz="UTC").timestamp()) for v in slot]
            for slot in intervals("2020-01-01")
        ]
        inputs["observation_time"][:] = inputs["slot_bounds"].mean(-1).astype("int64")[:, None]
        inputs["target_bounds"][:] = inputs["slot_bounds"][2]
        labels["water"][:] = 0.01 * index
        labels["snow_ice"][:] = 0.01 * index
        labels["land"][:] = 1 - 0.02 * index
        for key in (
            "water_valid",
            "snow_ice_valid",
            "land_valid",
            "soil_moisture_domain_valid",
            "soil_moisture_valid",
        ):
            labels[key][:] = 1
        labels["cover_known"][:] = 1
        labels["soil_moisture"][:] = 0.2 + 0.1 * index
        labels["soil_moisture_weight"][:] = 1
        for task in ("et_water", "et_ice", "et_land"):
            labels[f"{task}_native_valid"][0] = 1
            labels[f"{task}_native_value"][0] = index + 1
            labels[f"{task}_native_weights"][0] = 1 / 1600
        validate_arrays(inputs, labels)
        a, b = root / f"{index}_input.zarr", root / f"{index}_label.zarr"
        writer = SampleShardWriter(
            str(a),
            str(b),
            1,
            schema=schema,
            attrs=dict(config_sha256=fingerprint(cfg), input_grids=grid_metadata(sample)),
        )
        writer.write(inputs, labels)
        writer.finalize()
        rows.append(
            dict(
                sample_id=str(index),
                input_path=a.name,
                label_path=b.name,
                shard_index=0,
                version=VERSION,
                input_profile=PROFILE,
                split="train",
                frequency_bin=index,
                sampling_group=index,
                jrc_observed=True,
                target_time="2020-01-01",
                target_end="2020-01-09",
                climate_class=1,
                crs=sample["crs"],
                transform=json.dumps(sample["transform"]),
            )
        )
    manifest = root / "samples.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


@pytest.mark.parametrize("num_workers", [0, 2])
def test_strict_storage_roundtrip_statistics_and_config(tmp_path, num_workers):
    manifest = make_dataset(tmp_path)
    dataset = RetrievalDataset(manifest, "train")
    first = dataset[0]
    assert first["inputs"]["s1"].shape == (5, 2, 120, 120)
    torch.testing.assert_close(
        first["inputs"]["landsat"][0, 0, 0, :6],
        torch.tensor([0, 1, 1, 2, 3, 3], dtype=torch.float16),
    )
    flipped = RetrievalDataset(manifest, "train", training=True, flip_probability=1)[0]
    for key, value in first["inputs"].items():
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            torch.testing.assert_close(flipped["inputs"][key], value.flip(-1, -2))
    assert first["data_samples"]["soil_moisture_valid"].dtype == torch.bool
    assert first["data_samples"]["et_land_native_weights"].shape == (
        RetrievalSchema().native_capacity,
        40,
        40,
    )
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    configure_data_contract(cfg)
    cfg.train_dataloader.dataset.manifest = str(manifest)
    cfg.train_dataloader.dataset.data_root = str(tmp_path)
    stats = compute_retrieval_statistics(cfg, num_workers=num_workers)
    assert stats["tasks"]["soil_moisture"]["count"] == 2
    assert stats["tasks"]["soil_moisture"]["mean"] == pytest.approx(0.25)
    cfg.model.target_statistics = stats
    configure_target_statistics(cfg, training=True)
    assert cfg.val_evaluator.tasks == cfg.model.tasks
    table = pd.read_parquet(manifest)
    table.loc[0, "version"] = "old"
    table.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="version"):
        RetrievalDataset(manifest, "train")
