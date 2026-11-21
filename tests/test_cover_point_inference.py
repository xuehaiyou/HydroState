"""The point evaluator uses the actual label-free model path with all seven tasks."""

import torch
from retrieval.test_model import batch
from test_cosa import config
from test_mtlora import tiny  # noqa: F401

from hydrostate.inference.regional import predict_records
from hydrostate.registry import MODELS


def test_label_free_cover_inference_closes_and_preserves_invalid_pixels(tmp_path):
    model = MODELS.build(config(tmp_path).model).eval()
    inputs = batch()["inputs"]
    for key in list(inputs):
        if key.endswith("_valid") and inputs[key].ndim == 5:
            inputs[key][0, :, :, :3, :3] = False
    records = []
    for i in range(2):
        records.append(
            dict(
                inputs={
                    k: (
                        {t: v[i] for t, v in value.items()} if isinstance(value, dict) else value[i]
                    )
                    for k, value in inputs.items()
                }
            )
        )
    result = predict_records(model, records, "cpu")
    assert not result[0]["prediction_valid"][0, 0, 0]
    assert result[1]["prediction_valid"].all()
    for item in result:
        cover = torch.stack([item[t] for t in ("water", "snow_ice", "land")])
        assert torch.isfinite(cover).all()
        torch.testing.assert_close(cover.sum(0), torch.ones_like(item["water"]))
