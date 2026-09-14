import json

import numpy as np
import zarr

from hydrostate.storage import SampleShardWriter, Schema


def test_sample_shard_writer(tmp_path):
    schema = Schema(time_steps=2, height=8, width=8)
    input_path = tmp_path / "inputs.zarr"
    label_path = tmp_path / "labels.zarr"
    writer = SampleShardWriter(str(input_path), str(label_path), 1, schema=schema)
    inputs = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in schema.inputs.items()
    }
    targets = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in schema.targets.items()
    }
    assert writer.write(inputs, targets) == 0
    writer.finalize()
    assert json.loads((input_path / "_SUCCESS.json").read_text())["complete"]
    assert zarr.open_group(str(input_path), mode="r")["s2"].shape == (1, 2, 12, 8, 8)
