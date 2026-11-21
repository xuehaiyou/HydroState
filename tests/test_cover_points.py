"""Real invariants for point support, immutable acquisition and classification."""

import json

import numpy as np
import pandas as pd
import pytest

from hydrostate.data.cover_points.collection import collect
from hydrostate.data.cover_points.contract import CoverPointSchema
from hydrostate.data.cover_points.prepare import prepare
from hydrostate.data.cover_points.validation import audit
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.datasets.cover_point_dataset import CoverPointDataset
from hydrostate.evaluation.cover_point_metrics import classification_metrics, summarize


def source_file(tmp_path):
    p = tmp_path / "manual.csv"
    rows = []
    for j, value in enumerate([1, 2, 3, 4, 5]):
        rows.append(
            dict(
                site_name="Test",
                image_date="20210615",
                image_product="Sentinel-2_TOA",
                **{"class": value},
                geometry=f"MULTIPOINT ((-121.7 {46.8 + j * 0.0001}))",
            )
        )
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def fake_inputs(ee, sample, cache, norm, config):
    schema = CoverPointSchema()
    arrays = {k: np.zeros(v.tail_shape, dtype=v.dtype) for k, v in schema.inputs.items()}
    start = int(pd.Timestamp(sample["target_time"], tz="UTC").timestamp())
    end = int(pd.Timestamp(sample["target_end"], tz="UTC").timestamp())
    arrays["target_bounds"][:] = [start, end]
    arrays["slot_bounds"][:] = [
        [start + d * 86400, start + (d + 8) * 86400] for d in [-16, -8, 0, 8, 16]
    ]
    arrays["sensor_valid"][:] = 1
    arrays["observation_time"][:] = arrays["slot_bounds"].mean(axis=1)[:, None]
    for name in INPUT_GROUPS:
        arrays[name][:] = 0.25
        arrays[f"{name}_valid"][:] = 1
        arrays[f"{name}_quality"][:, 0] = 1
    return arrays, {}, True


def test_roundtrip_resume_publication_and_variable_points(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    root = tmp_path / "data"
    report = prepare(source, root)
    assert report["points"] == 5
    assert prepare(source, root) == report
    monkeypatch.setattr("hydrostate.data.cover_points.collection.input_arrays", fake_inputs)
    with pytest.raises(ValueError, match="pilot"):
        collect(root, project="test", ee=object(), norm={})
    collect(root, project="test", pilot=True, ee=object(), norm={})
    result = audit(root, pilot=True)
    assert result["input_supported_points"] == 5

    # Resume must not call the network again.
    def unexpected(*args, **kwargs):
        raise AssertionError("Committed data downloaded again")

    monkeypatch.setattr("hydrostate.data.cover_points.collection.input_arrays", unexpected)
    collect(root, project="test", ee=object(), norm={})
    audit(root, publish=True)
    dataset = CoverPointDataset(root)
    item = dataset[0]
    assert item["inputs"]["s1"].shape[-2:] == (120, 120)
    assert item["points"]["cover_point_valid"].all()
    assert item["points"]["cover_point_class"].tolist() == [1, 1, 1, 2, 0]
    assert (item["points"]["cover_point_rc"] >= 5).all()
    assert (root / "_PUBLISHED.json").exists()
    with pytest.raises(ValueError, match="immutable"):
        collect(root, project="test", ee=object(), norm={})


def test_conflicting_duplicate_and_binary_source_rejected(tmp_path):
    source = source_file(tmp_path)
    rows = pd.read_csv(source)
    conflict = rows.iloc[[0]].copy()
    conflict["class"] = 5
    pd.concat([rows, conflict]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(source, tmp_path / "bad")
    rows.loc[0, "class"] = 0
    rows.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Unknown"):
        prepare(source, tmp_path / "binary")


def test_metrics_support_conflicts_and_undefined_classes():
    frame = pd.DataFrame(
        dict(
            cover_class=[0, 1, 2, 2],
            predicted_class=[0, 2, 2, -1],
            label_valid=[True] * 4,
            prediction_valid=[True, True, True, False],
            training_overlap=[False] * 4,
            sample_id=["a"] * 4,
            observation_date=["2021-06-15"] * 4,
            output_row=[1, 2, 2, 3],
            output_col=[1] * 4,
            site_name=["Test"] * 4,
        )
    )
    result = summarize(frame)
    assert result["point_metrics"]["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert result["conflicting_cells"] == 1
    assert result["consistent_unique_cell_metrics"]["count"] == 1
    assert result["invalid_prediction_points"] == 1
    frame["training_overlap"] = True
    empty = summarize(frame)
    assert empty["point_metrics"]["count"] == 0
    assert classification_metrics(frame.iloc[:0])["macro_f1"] is None
    json.dumps(empty, allow_nan=False)


def test_many_points_in_same_pixel_do_not_become_area_fractions(tmp_path):
    root = tmp_path / "data"
    prepare(source_file(tmp_path), root)
    points = pd.read_parquet(root / "points.parquet")
    assert points.point_index.is_unique
    assert len(points[["sample_id", "output_row", "output_col"]].drop_duplicates()) < len(points)
    assert set(points.cover_class) == {0, 1, 2}


def test_rejected_input_is_not_success_or_silent_zero_label(tmp_path, monkeypatch):
    root = tmp_path / "data"
    prepare(source_file(tmp_path), root)

    def missing(*args):
        arrays, scenes, _ = fake_inputs(*args)
        for name in INPUT_GROUPS:
            arrays[f"{name}_valid"][:] = 0
        return arrays, scenes, False

    monkeypatch.setattr("hydrostate.data.cover_points.collection.input_arrays", missing)
    result = collect(root, project="test", pilot=True, ee=object(), norm={})
    assert result["accepted"] == 0
    assert result["rejected"] > 0
    assert not (root / "_PUBLISHED.json").exists()
    assert len(pd.read_parquet(root / "points.parquet")) == 5


def test_network_failure_is_resumable_not_rejection(tmp_path, monkeypatch):
    root = tmp_path / "data"
    prepare(source_file(tmp_path), root)

    def failed(*args):
        raise TimeoutError("simulated network interruption")

    monkeypatch.setattr("hydrostate.data.cover_points.collection.input_arrays", failed)
    with pytest.raises(TimeoutError):
        collect(root, project="test", pilot=True, ee=object(), norm={})
    assert not list((root / "rejected").glob("*.json"))
    monkeypatch.setattr("hydrostate.data.cover_points.collection.input_arrays", fake_inputs)
    collect(root, project="test", pilot=True, ee=object(), norm={})
    assert audit(root, pilot=True)["passed"]
