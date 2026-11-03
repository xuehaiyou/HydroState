import json

import numpy as np
import pandas as pd
import torch

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.retrieval.alignment import grid_metadata
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.data.site.contract import VERSION, SiteSchema
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.storage import SampleShardWriter


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
