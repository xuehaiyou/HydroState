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


def test_manual_cover_real_model_gradient_and_export(tmp_path):
    cfg = config(tmp_path)
    cfg.model.manual_cover_weight = 2.0
    model = MODELS.build(cfg.model)
    raw = batch()
    samples = raw["data_samples"]
    for key in samples:
        if key.endswith("_valid"):
            samples[key].zero_()
    for name in ("water", "snow_ice", "land"):
        samples["manual_" + name] = torch.zeros_like(samples["water"])
        samples["manual_" + name + "_valid"] = torch.zeros_like(samples["water"], dtype=torch.bool)
    samples["manual_water"][0] = 1
    samples["manual_water_valid"][0] = True
    samples["manual_snow_ice"][1] = 1
    for name in ("water", "snow_ice", "land"):
        samples["manual_" + name + "_valid"][1] = True
    losses = model.loss(raw["inputs"], samples)
    assert losses["loss_manual_cover"] > 0
    assert all(v == 0 for k, v in losses.items() if k.startswith("valid_count_"))
    losses["loss_manual_cover"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    predictions = model.eval().predict(raw["inputs"], samples)
    assert predictions[0]["manual_water_valid"].all()
