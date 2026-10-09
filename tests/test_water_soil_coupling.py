"""Physical support, gradient isolation and prediction contracts for WF -> SM."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hydrostate.engine.visualization_hook import (
    HydroVisualizationHook,
    _cpu_sample,
    comparison_maps,
)
from hydrostate.evaluation.hydro_metrics import HydroStateMetric
from hydrostate.losses.uncertainty_nll import UncertaintyWeightedNLL
from hydrostate.models.hydrostate_model import (
    HydroStateModel,
    combined_supervision_values,
    prepare_supervision,
)
from hydrostate.models.water_soil_coupling import WaterSoilCoupling
from hydrostate.normalization import TASKS


def example():
    water = torch.tensor([[[[1.0, 0.5], [0.0, 1.0]]]])
    outputs = dict(
        water=water.clone().requires_grad_(),
        soil_moisture=torch.tensor([[[[0.8, 0.2], [0.4, 0.9]]]], requires_grad=True),
        et=torch.full_like(water, 3.0, requires_grad=True),
    )
    samples = dict(
        water=water, water_valid=torch.ones_like(water, dtype=torch.bool),
        soil_moisture=torch.tensor([[[[1 / 3]]]]),
        soil_moisture_valid=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        prediction_valid=torch.ones_like(water, dtype=torch.bool),
        et_native_weights=torch.full_like(water, 0.25),
        et_native_value=torch.tensor([[3.0]]), et_native_valid=torch.tensor([[True]]),
        metainfo=dict(sample_id=["mixed"], target_time=["2026-01-01"]),
    )
    return outputs, samples


def probe(outputs, coupling, statistics=None):
    model = SimpleNamespace(
        _forward=lambda inputs: outputs, water_soil_coupling=coupling,
    )
    if statistics is not None:
        model.loss_module = UncertaintyWeightedNLL(statistics)
    return model


@pytest.mark.parametrize("source", ["observed", "predicted"])
def test_land_mean_and_gradients_exclude_water_without_shrinking_sm(source):
    outputs, samples = example()
    # Invalid prediction values on open water must not contaminate the reduction.
    outputs["soil_moisture"].data[0, 0, 0, 0] = float("nan")
    support = WaterSoilCoupling(supervision_source=source).prepare(
        outputs, samples, supervision=True,
    )
    mean = support.project(outputs["soil_moisture"])
    torch.testing.assert_close(mean, torch.tensor([[[[1 / 3]]]]))
    assert support.valid.all()
    mean.sum().backward()
    torch.testing.assert_close(
        outputs["soil_moisture"].grad, torch.tensor([[[[0.0, 1 / 3], [2 / 3, 0.0]]]]),
    )
    assert outputs["water"].grad is None


def test_uniform_land_sm_is_unchanged_by_water_fraction():
    outputs, samples = example()
    outputs["soil_moisture"] = torch.full_like(outputs["water"], 0.25)
    support = WaterSoilCoupling().prepare(outputs, samples, supervision=True)
    torch.testing.assert_close(support.project(outputs["soil_moisture"]),
                               torch.tensor([[[[0.25]]]]))


def test_disabled_threshold_does_not_shrink_sm_for_tiny_positive_land_area():
    outputs, samples = example()
    samples["water"].fill_(1 - torch.finfo(torch.float32).eps)
    outputs["soil_moisture"] = torch.full_like(outputs["water"], 0.25)
    support = WaterSoilCoupling(min_land_fraction=0, min_window_land_fraction=0).prepare(
        outputs, samples, supervision=True,
    )
    assert support.valid.all()
    torch.testing.assert_close(support.project(outputs["soil_moisture"]),
                               torch.tensor([[[[0.25]]]]))


def test_bfloat16_prediction_uses_float32_reduction():
    outputs, samples = example()
    outputs = {k: v.detach().to(torch.bfloat16).requires_grad_() for k, v in outputs.items()}
    support = WaterSoilCoupling().prepare(outputs, samples, supervision=True)
    pooled = support.project(outputs["soil_moisture"])
    sm = outputs["soil_moisture"].float()
    assert pooled.dtype == torch.float32
    expected = (sm[0, 0, 0, 1] / 3 + sm[0, 0, 1, 0] * 2 / 3)[None]
    torch.testing.assert_close(pooled.flatten(), expected)
    pooled.sum().backward()
    assert torch.isfinite(outputs["soil_moisture"].grad).all()


@pytest.mark.parametrize("invalid", ["water", "unknown", "unobserved", "too_little_land"])
def test_empty_or_unknown_land_has_zero_sm_losses_and_finite_gradients(
    invalid, target_statistics,
):
    outputs, samples = example()
    if invalid == "water":
        samples["water"].fill_(1)
    elif invalid == "unknown":
        samples["water"][0, 0, 0, 1] = float("nan")
        samples["water_valid"][0, 0, 0, 1] = False
    elif invalid == "unobserved":
        samples["prediction_valid"].zero_()
    else:
        samples["water"].fill_(1)
        samples["water"][0, 0, 0, 0] = 0.98  # 0.5% land over the window.
    samples["soil_moisture"].fill_(float("nan"))
    model = probe(outputs, WaterSoilCoupling(), target_statistics)
    auxiliary = {t: v.detach().clone().requires_grad_() for t, v in outputs.items()}
    losses = HydroStateModel._loss_from_outputs(
        model, outputs, samples, auxiliary, dict.fromkeys(TASKS, 0.2),
    )
    assert losses["valid_count_soil_moisture"] == 0
    total = losses["loss_soil_moisture"] + losses["loss_aux_soil_moisture"]
    assert total == 0
    total.backward()
    for sm in (outputs["soil_moisture"], auxiliary["soil_moisture"]):
        assert torch.isfinite(sm.grad).all() and (sm.grad == 0).all()


def test_optional_soil_domain_excludes_non_soil_without_dropping_valid_land():
    outputs, samples = example()
    samples["soil_moisture_domain_valid"] = samples["prediction_valid"].clone()
    samples["soil_moisture_domain_valid"][0, 0, 1, 0] = False
    support = WaterSoilCoupling().prepare(outputs, samples, supervision=True)
    assert support.valid.all()
    torch.testing.assert_close(support.project(outputs["soil_moisture"]),
                               torch.tensor([[[[0.2]]]]))


@pytest.mark.parametrize("water_only_station", [False, True])
def test_station_and_auxiliary_share_the_same_land_support(water_only_station):
    outputs, samples = example()
    samples.update(site_value=torch.tensor([[0.2, 0.3, 3.0]]),
                   site_valid=torch.ones(1, 3, dtype=torch.bool),
                   site_footprint=torch.full_like(outputs["water"], 0.25))
    if water_only_station:
        samples["site_footprint"].zero_()
        samples["site_footprint"][0, 0, 0, 0] = 1
    support = WaterSoilCoupling().prepare(outputs, samples, supervision=True)
    plans = prepare_supervision(outputs, samples, support)
    for task in TASKS:
        expected, target, valid = combined_supervision_values(
            task, outputs[task], samples, support,
        )
        torch.testing.assert_close(plans[task].project(outputs[task]), expected)
        torch.testing.assert_close(plans[task].target, target)
        torch.testing.assert_close(plans[task].valid, valid)
    assert plans["soil_moisture"].count == (1 if water_only_station else 2)
    assert plans["et"].count == 2
    assert plans["water"].count == 5


def test_auxiliary_uses_final_detached_wf_and_preserves_wf_et_losses(target_statistics):
    outputs, samples = example()
    coupling = WaterSoilCoupling(supervision_source="predicted")
    model = probe(outputs, coupling, target_statistics)
    auxiliary = {t: v.detach().clone().requires_grad_() for t, v in outputs.items()}
    auxiliary["water"].data.zero_()  # Must not change the auxiliary SM support.
    auxiliary["soil_moisture"].data.fill_(0.1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        losses = HydroStateModel._loss_from_outputs(
            model, outputs, samples, auxiliary, dict.fromkeys(TASKS, 0.2),
        )
        losses["loss_aux_soil_moisture"].backward()
    operations = {event.key for event in profile.key_averages()}
    assert not operations & {"aten::masked_select", "aten::nonzero", "aten::_local_scalar_dense"}
    expected = 0.2 * ((0.1 - 1 / 3) / target_statistics["tasks"]["soil_moisture"]["std"]) ** 2
    torch.testing.assert_close(losses["loss_aux_soil_moisture"], torch.tensor(expected))
    assert outputs["water"].grad is None and auxiliary["water"].grad is None
    baseline = HydroStateModel._loss_from_outputs(probe(outputs, None, target_statistics),
                                                outputs, samples)
    for task in ("water", "et"):
        torch.testing.assert_close(losses[f"loss_{task}"], baseline[f"loss_{task}"])


def test_missing_wf_outside_station_does_not_invalidate_station():
    outputs, samples = example()
    samples["water_valid"][0, 0, 0, 0] = False
    samples.update(site_value=torch.tensor([[0.2, 0.3, 3.0]]),
                   site_valid=torch.ones(1, 3, dtype=torch.bool),
                   site_footprint=torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]))
    support = WaterSoilCoupling().prepare(outputs, samples, supervision=True)
    plan = prepare_supervision(outputs, samples, support)["soil_moisture"]
    assert plan.valid.tolist() == [False, True]
    torch.testing.assert_close(plan.project(outputs["soil_moisture"])[1], torch.tensor(0.2))


def test_prediction_nodata_uses_predicted_wf_but_metrics_use_observed_support(target_statistics):
    outputs, samples = example()
    # Deliberately wrong WF: prediction is masked but the SM validation label
    # must remain in the evaluator, rather than rewarding mask manipulation.
    outputs["water"].data.fill_(1)
    model = probe(outputs, WaterSoilCoupling())
    item = HydroStateModel.predict(model, {}, samples)[0]
    assert torch.isnan(item["soil_moisture"]).all()
    assert not item["soil_moisture_prediction_valid"].any()
    assert item["valid_soil_moisture"].all()
    torch.testing.assert_close(item["supervised_soil_moisture"], torch.tensor([[[1 / 3]]]))
    metric = HydroStateMetric(target_statistics)
    metric.process(None, [item])
    metrics = metric.compute_metrics(metric.results)
    assert metrics["soil_moisture_rmse"] < 1e-6
    assert metrics["soil_moisture_observation_count"] == 1
    assert metrics["soil_moisture_prediction_coverage"] == 0
    fine, mean, target = comparison_maps(_cpu_sample(item), "soil_moisture")
    assert np.isnan(fine).all() and np.isfinite(mean).all() and np.isfinite(target).all()
    assert torch.isfinite(outputs["soil_moisture"]).all()  # No in-place mutation.
    assert torch.isfinite(item["water"]).all() and torch.isfinite(item["et"]).all()


def test_inference_without_labels_preserves_mixed_pixel_sm_and_masks_near_pure_water():
    outputs, _ = example()
    outputs["water"].data[0, 0, 0, 0] = 0.995
    item = HydroStateModel.predict(probe(outputs, WaterSoilCoupling()), {})[0]
    assert torch.isnan(item["soil_moisture"][0, 0, 0])
    assert torch.isnan(item["soil_moisture"][0, 1, 1])
    torch.testing.assert_close(item["soil_moisture"][0, 0, 1], torch.tensor(0.2))
    torch.testing.assert_close(item["soil_moisture_land_fraction"][0, 0, 1], torch.tensor(0.5))


def test_disabled_coupling_retains_legacy_outputs(target_statistics):
    outputs, samples = example()
    old = probe(outputs, None, target_statistics)
    disabled = probe(outputs, WaterSoilCoupling(enabled=False), target_statistics)
    for task in TASKS:
        torch.testing.assert_close(HydroStateModel.predict(old, {}, samples)[0][task],
                                   HydroStateModel.predict(disabled, {}, samples)[0][task])
    expected = HydroStateModel._loss_from_outputs(old, outputs, samples)
    actual = HydroStateModel._loss_from_outputs(disabled, outputs, samples)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_coupled_prediction_renders_with_explicit_mask_and_weight_labels(tmp_path):
    outputs, samples = example()
    item = HydroStateModel.predict(probe(outputs, WaterSoilCoupling()), {}, samples)[0]
    hook = HydroVisualizationHook()
    hook._label = "epoch_001"
    path = tmp_path / "mixed.png"
    hook._draw(_cpu_sample(item), path)
    assert path.stat().st_size > 1000


@pytest.mark.parametrize("config", [dict(supervision_source="bad"), dict(enabled="yes"),
                                   dict(min_land_fraction=-0.1), dict(min_land_fraction=1),
                                   dict(min_window_land_fraction=float("nan"))])
def test_invalid_config_fails_early(config):
    with pytest.raises(ValueError):
        WaterSoilCoupling(**config)
