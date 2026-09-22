import torch

from hydrostate.losses import MaskedHuberLoss


def test_masked_huber_ignores_invalid_values():
    prediction = torch.tensor([[[[1.0, 100.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[[[True, False]]]])
    loss = MaskedHuberLoss(delta=1.0)(prediction, target, valid)
    assert torch.isclose(loss, torch.tensor(0.5))
    loss.backward()
    assert prediction.grad[0, 0, 0, 1] == 0


def test_empty_mask_returns_differentiable_zero():
    prediction = torch.ones((2, 1, 4, 4), requires_grad=True)
    loss = MaskedHuberLoss()(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction))
    assert loss.item() == 0
    loss.backward()
    assert prediction.grad is not None


def test_native_et_preserves_subpixel_variation_and_excludes_partial_support():
    from hydrostate.models.hydrostate_model import supervision_values

    pred = torch.tensor([[[[1.0, 3.0], [10.0, 20.0]]]], requires_grad=True)
    samples = dict(
        et_native_weights=torch.tensor([[[[0.5, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.5, 0.5]]]]),
        et_native_value=torch.tensor([[2.0, 0.0]]),
        et_native_valid=torch.tensor([[True, True]]),
        prediction_valid=torch.tensor([[[[True, True], [True, False]]]]),
    )
    pooled, target, valid = supervision_values("et", pred, samples)
    assert valid.tolist() == [[True, False]]
    loss = MaskedHuberLoss()(pooled, target, valid)
    assert loss.item() == 0  # [1,3] need not individually equal the coarse target 2.
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_smap_window_mean_uses_only_observed_predictions():
    from hydrostate.models.hydrostate_model import supervision_values

    pred = torch.tensor([[[[0.1, 0.3], [0.9, 0.9]]]], requires_grad=True)
    samples = dict(
        soil_moisture=torch.tensor([[[[0.2]]]]),
        soil_moisture_valid=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        prediction_valid=torch.tensor([[[[True, True], [False, False]]]]),
    )
    pooled, target, valid = supervision_values("soil_moisture", pred, samples)
    torch.testing.assert_close(pooled, target)
    assert valid.all()
    samples["prediction_valid"].zero_()
    pooled, target, valid = supervision_values("soil_moisture", pred, samples)
    loss = MaskedHuberLoss()(pooled, target, valid)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(pred.grad).all()


def test_metrics_evaluate_native_support_and_singleton_r2():
    import math

    from hydrostate.evaluation.hydro_metrics import HydroStateMetric

    metric = HydroStateMetric()
    sample = {}
    for task in ("water", "soil_moisture", "et"):
        sample[task] = torch.full((1, 40, 40), 100.0)  # Fine map is not the metric support.
        sample[f"supervised_{task}"] = torch.tensor([2.0])
        sample[f"target_{task}"] = torch.tensor([1.0])
        sample[f"valid_{task}"] = torch.tensor([True])
    metric.process(None, [sample])
    result = metric.compute_metrics(metric.results)
    assert result["et_rmse"] == 1.0
    assert math.isnan(result["soil_moisture_r2"])
