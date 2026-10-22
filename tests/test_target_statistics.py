import copy

import numpy as np
import pandas as pd
import pytest
import zarr
from mmengine.config import Config

from hydrostate.cli import configure_target_statistics
from hydrostate.normalization import load_target_statistics, training_fingerprint
from hydrostate.storage import SampleShardWriter
from hydrostate.legacy.schema import LegacySchema as Schema
from hydrostate.statistics import RunningMoments
from hydrostate.legacy.statistics import compute_statistics


def test_population_moments_and_degenerate_labels():
    moments = RunningMoments()
    moments.update([1.0, 2.0])
    moments.update([7.0])
    np.testing.assert_allclose(moments.result()["std"], np.std([1.0, 2.0, 7.0], ddof=0))
    with pytest.raises(ValueError, match="Nonfinite"):
        moments.update([float("nan")])
    constant = RunningMoments()
    constant.update([2.0, 2.0])
    with pytest.raises(ValueError, match="Near-constant"):
        constant.result()


def test_statistics_use_only_training_and_actual_observed_support(tmp_path):
    schema = Schema(time_steps=1, height=6, width=6)
    input_path, label_path = tmp_path / "inputs.zarr", tmp_path / "labels.zarr"
    writer = SampleShardWriter(str(input_path), str(label_path), 4, schema=schema)
    for index in range(4):
        inputs = {
            name: np.zeros(spec.tail_shape, dtype=spec.dtype)
            for name, spec in schema.inputs.items()
        }
        targets = {
            name: np.zeros(spec.tail_shape, dtype=spec.dtype)
            for name, spec in schema.targets.items()
        }
        inputs["s1_valid"][:] = True
        inputs["sensor_valid"][:] = True
        if index == 1:
            inputs["s1_valid"][..., 3:, 3:] = False
        targets["water"][:] = index if index < 2 else 100
        targets["soil_moisture"][:] = [0.1, 0.3, 100.0, 200.0][index]
        targets["water_valid"][:] = True
        targets["soil_moisture_valid"][:] = True
        if index == 0:
            targets["water_valid"][..., 0, 0] = False
        writer.write(inputs, targets)
    writer.finalize()
    group = zarr.open_group(str(label_path), mode="a")
    for index in range(4):
        native = group.require_group(f"et_native/{index}")
        weights = np.zeros((2, 2, 2), dtype=np.float32)
        weights[0, 0, 0], weights[1, 1, 1] = 1.0, 1.0
        native.create_dataset("weights", data=weights)
        native.create_dataset(
            "value", data=np.array([1.0, 3.0] if index == 0 else [5.0, 999.0], dtype=np.float32)
        )
        native.create_dataset("valid", data=np.ones(2, dtype=bool))
    table = pd.DataFrame(
        dict(
            sample_id=["a", "b", "c", "d"],
            split=["train", "train", "validation", "test"],
            input_shard=["inputs.zarr"] * 4,
            label_shard=["labels.zarr"] * 4,
            input_row=range(4),
            label_row=range(4),
        )
    )
    manifest = tmp_path / "samples.parquet"
    table.to_parquet(manifest, index=False)
    report = compute_statistics(manifest, tmp_path, num_workers=0, batch_size=2)
    assert report["training_samples"] == 2
    assert report["tasks"]["water"]["count"] == 6
    assert report["tasks"]["water"]["mean"] == 0.5
    assert report["tasks"]["water"]["std"] == 0.5
    np.testing.assert_allclose(report["tasks"]["soil_moisture"]["std"], 0.1)
    assert report["tasks"]["soil_moisture"]["count"] == 2
    assert report["tasks"]["et"]["count"] == 3
    assert report["tasks"]["et"]["mean"] == 3.0
    np.testing.assert_allclose(report["tasks"]["et"]["std"], np.std([1.0, 3.0, 5.0]))
    changed = table.copy()
    changed.loc[changed.split != "train", "sample_id"] = ["new-val", "new-test"]
    assert training_fingerprint(changed.iloc[::-1]) == training_fingerprint(table)


def test_training_config_freezes_one_statistics_version(target_statistics, tmp_path):
    table = pd.DataFrame(dict(sample_id=["train-a", "val-a"], split=["train", "validation"]))
    manifest = tmp_path / "samples.parquet"
    table.to_parquet(manifest, index=False)
    statistics = copy.deepcopy(target_statistics)
    statistics["training_fingerprint"] = training_fingerprint(table)
    cfg = Config(
        dict(
            model=dict(target_statistics=statistics),
            train_dataloader=dict(dataset=dict(manifest=str(manifest), split="train")),
            val_evaluator=dict(type="HydroStateMetric"),
            test_evaluator=dict(type="HydroStateMetric"),
        )
    )
    configure_target_statistics(cfg, training=True)
    assert cfg.val_evaluator.target_statistics == cfg.model.target_statistics
    assert cfg.test_evaluator.target_statistics == cfg.model.target_statistics
    table.loc[0, "sample_id"] = "different-training-sample"
    table.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="manifest differs"):
        configure_target_statistics(cfg, training=True)
    configure_target_statistics(cfg, training=False)  # Inference reuses original training moments.


def test_statistics_reject_heldout_origin(target_statistics):
    target_statistics["split"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        load_target_statistics(target_statistics)
