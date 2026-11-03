import json

import numpy as np
import pandas as pd
import pytest
import torch

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.alignment import grid_metadata
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.data.site.contract import VERSION, SiteSchema
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.storage import SampleShardWriter


def test_jrc_never_water_is_distinct_from_no_observations():
    from hydrostate.data.site.pilot import observed_occurrence

    assert observed_occurrence(dict(occurrence=-1, valid_obs=445, detections=0)) == 0
    assert observed_occurrence(dict(occurrence=80, valid_obs=100, detections=90)) == 80
    with pytest.raises(ValueError, match="no JRC observations"):
        observed_occurrence(dict(occurrence=-1, valid_obs=0, detections=0))
    with pytest.raises(ValueError, match="water detections"):
        observed_occurrence(dict(occurrence=-1, valid_obs=100, detections=5))


def test_independent_station_zarr_and_support_augmentation(tmp_path):
    schema = SiteSchema()
    config = dict(stage="test_fixture")
    (tmp_path / "collection.json").write_text(
        json.dumps(dict(config, version=VERSION, input_profile=PROFILE))
    )
    sample = dict(crs="EPSG:32635", transform=[10, 0, 100, 0, -10, 100])
    inputs = {key: np.zeros(spec.tail_shape, spec.dtype) for key, spec in schema.inputs.items()}
    targets = {key: np.zeros(spec.tail_shape, spec.dtype) for key, spec in schema.targets.items()}
    targets["site_sm_native_value"][0] = 0.23
    targets["site_sm_native_valid"][0] = 1
    targets["site_sm_native_weights"][0, 20, 20] = 1
    writer = SampleShardWriter(
        str(tmp_path / "inputs.zarr"),
        str(tmp_path / "labels.zarr"),
        1,
        schema,
        attrs=dict(config_sha256=fingerprint(config), input_grids=grid_metadata(sample)),
    )
    writer.write(inputs, targets)
    writer.finalize()
    table = pd.DataFrame(
        [
            dict(
                sample_id="fixture",
                split="train",
                version=VERSION,
                input_path="inputs.zarr",
                label_path="labels.zarr",
                shard_index=0,
                observation_capacity=1,
                target_time="2021-01-01",
                sampling_group=0,
                frequency_bin=0,
                climate_class=1,
                crs=sample["crs"],
                transform=json.dumps(sample["transform"]),
            )
        ]
    )
    manifest = tmp_path / "samples.parquet"
    table.to_parquet(manifest, index=False)
    item = SiteDataset(manifest, "train")[0]
    assert item["inputs"]["s1"].shape == (5, 2, 120, 120)
    assert not item["data_samples"]["soil_moisture_valid"].any()
    assert item["data_samples"]["site_sm_native_valid"].all()
    flipped = SiteDataset(manifest, "train", training=True, flip_probability=1)[0]
    assert flipped["data_samples"]["site_sm_native_weights"][0, 19, 19] == 1
    batch = hydro_collate([item, flipped])
    torch.testing.assert_close(
        batch["data_samples"]["site_sm_native_value"], torch.tensor([[0.23], [0.23]])
    )


def test_multi_station_collection_preserves_committed_manifest_and_split(tmp_path, monkeypatch):
    from hydrostate.data.site import pilot

    schema = SiteSchema()
    inputs = {key: np.zeros(spec.tail_shape, spec.dtype) for key, spec in schema.inputs.items()}
    for name in inputs:
        if name.endswith("_valid"):
            inputs[name][...] = 1
    monkeypatch.setattr(pilot, "input_arrays", lambda *args: (inputs, {}, True))
    monkeypatch.setattr(
        pilot, "strata", lambda *args: dict(frequency_bin=0, climate_class=8, landcover=10)
    )
    first = dict(
        site_id="a",
        group_id="a",
        lon=25.0,
        lat=41.0,
        target_time="2021-01-01",
        target_end="2021-01-09",
        sm=dict(value=0.2, valid=True),
        split="train",
        candidate_id="a1",
    )
    second = dict(first, site_id="b", group_id="b", lon=26.0, split="validation", candidate_id="b1")
    config = dict(climate_asset="fixture", min_input_fraction=0.5, min_active_slots=2)
    assert len(pilot.collect_pilot(None, [first], tmp_path, config, {})) == 1
    assert len(pilot.collect_pilot(None, [second], tmp_path, config, {})) == 2
    assert len(pilot.collect_pilot(None, [first], tmp_path, config, {})) == 2
    frame = pd.read_parquet(tmp_path / "samples.parquet")
    assert set(frame.split) == {"train", "validation"}
    assert set(frame.candidate_id) == {"a1", "b1"}
