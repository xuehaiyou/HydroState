"""Independent references for CPU temporal metadata and fixed-shape reductions."""

import copy
import math
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from hydrostate.losses.uncertainty_nll import (
    UncertaintyWeightedNLL,
    global_observation_means,
    masked_standardized_mse,
)
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import (
    HydroStateModel,
    combined_supervision_values,
    prepare_supervision,
)
from hydrostate.models.olmoearth_encoder import INPUT_KEYS, MODALITIES, OlmoEarthEncoder
from hydrostate.models.temporal_metadata import prepare_temporal_metadata
from hydrostate.normalization import TASKS


def temporal_inputs(mask_channels=True):
    torch.manual_seed(72)
    valid = torch.tensor([
        [[1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    ], dtype=torch.bool)
    # Deliberately nonchronological: ordering must be preserved for rslearn.
    days = torch.tensor([48, 0, 32, 16]).reshape(1, 4, 1)
    times = (1600000000 + days * 86400 + torch.arange(3) * 86400).expand(2, -1, -1).clone()
    times[~valid] = torch.iinfo(torch.int64).min  # Must never convert invalid dates.
    inputs = dict(sensor_valid=valid, observation_time=times)
    for key, channels in zip(INPUT_KEYS, (2, 12, 11)):
        inputs[key] = torch.randn(2, 4, channels, 6, 6, requires_grad=True)
        shape = (2, 4, 1, 6, 6) if mask_channels else (2, 4, 6, 6)
        inputs[f"{key}_valid"] = torch.rand(shape) > 0.2
    return inputs


def legacy_context(inputs):
    from rslearn.train.model_context import ModelContext, RasterImage

    samples = []
    for batch in range(inputs["sensor_valid"].shape[0]):
        sample = {}
        for index, (key, modality) in enumerate(zip(INPUT_KEYS, MODALITIES)):
            valid = inputs["sensor_valid"][batch, :, index]
            if not torch.any(valid):
                continue
            value = inputs[key][batch, valid]
            if f"{key}_valid" in inputs:
                mask = inputs[f"{key}_valid"][batch, valid]
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                value = value * mask.to(value.dtype)
            sample[modality] = RasterImage(
                image=value.permute(1, 0, 2, 3).contiguous(),
                timestamps=OlmoEarthEncoder._timestamps_for_sample(
                    inputs["observation_time"][batch], valid, index
                ),
            )
        samples.append(sample)
    return ModelContext(inputs=samples, metadatas=[])


class CaptureBackbone(nn.Module):
    def forward(self, context):
        self.context = context
        features = torch.stack([
            sum(value.image.mean(dim=(0, 1)) for value in sample.values())
            for sample in context.inputs
        ])[:, None]
        return SimpleNamespace(feature_maps=[features])


def capture_encoder():
    encoder = OlmoEarthEncoder.__new__(OlmoEarthEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = CaptureBackbone()
    encoder.out_channels = 1
    return encoder


@pytest.mark.parametrize("mask_channels", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_temporal_gather_matches_context_alignment_and_input_gradients(mask_channels, prepared):
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth

    original = temporal_inputs(mask_channels)
    optimized = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in original.items()}
    if prepared:
        optimized["_temporal_metadata"], optimized["_temporal_indices"] = prepare_temporal_metadata(
            optimized["sensor_valid"], optimized["observation_time"]
        )
    reference = legacy_context(original)
    encoder = capture_encoder()
    expected = CaptureBackbone()(reference).feature_maps[0]
    actual = encoder(optimized)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(reference.inputs, encoder.backbone.context.inputs):
        assert left.keys() == right.keys()
        for modality in left:
            assert left[modality].timestamps == right[modality].timestamps
            torch.testing.assert_close(left[modality].image, right[modality].image, rtol=0, atol=0)
    wrapper = SimpleNamespace(
        timestamp_error_tolerance=timedelta(days=15),
        datetimes_to_timestamps=OlmoEarth.datetimes_to_timestamps,
    )
    a, am, _ = OlmoEarth._prepare_modality_inputs(wrapper, reference)
    b, bm, _ = OlmoEarth._prepare_modality_inputs(wrapper, encoder.backbone.context)
    assert am == bm
    for field in ["timestamps", *am, *(f"{modality}_mask" for modality in am)]:
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for key in INPUT_KEYS:
        torch.testing.assert_close(original[key].grad, optimized[key].grad, rtol=0, atol=0)


def test_preprocessor_keeps_temporal_controls_out_of_device_transfer():
    class InspectPreprocessor(HydroDataPreprocessor):
        def cast_data(self, value):
            if isinstance(value, dict) and "inputs" in value:
                self.transfer_keys = set(value["inputs"])
            return super().cast_data(value)

    inputs = temporal_inputs()
    targets = dict(water=torch.zeros(2, 1, 6, 6),
                   water_valid=torch.ones(2, 1, 6, 6, dtype=torch.bool),
                   soil_moisture=torch.zeros(2, 1, 1, 1))
    processor = InspectPreprocessor()
    result = processor(dict(inputs=inputs, data_samples=targets))["inputs"]
    assert "sensor_valid" not in processor.transfer_keys
    assert "observation_time" not in processor.transfer_keys
    assert "_temporal_indices" in processor.transfer_keys
    assert result["sensor_valid"].device.type == result["observation_time"].device.type == "cpu"
    assert "_temporal_metadata" not in inputs  # No mutation of the input dictionary.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        capture_encoder()(result)
    operations = {event.key for event in profile.key_averages()}
    assert not operations & {"aten::nonzero", "aten::_local_scalar_dense", "aten::masked_select"}


def test_all_missing_modalities_fail_before_device_transfer():
    with pytest.raises(ValueError, match="sample 0 has no valid satellite modality"):
        prepare_temporal_metadata(torch.zeros(1, 4, 3, dtype=torch.bool), torch.zeros(1, 4, 3))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("log_variance", [-20.0, math.log(1e-6), 0.7])
@pytest.mark.parametrize("all_missing", [False, True])
def test_fixed_shape_nll_and_auxiliary_match_selected_reference(
    target_statistics, task, log_variance, all_missing,
):
    module = UncertaintyWeightedNLL(target_statistics)
    module.log_variances[task].data.fill_(log_variance)
    legacy = copy.deepcopy(module)
    valid = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    if all_missing:
        valid.zero_()
    x = torch.tensor([[0.1, float("nan"), 0.9], [float("inf"), 0.2, float("nan")]],
                     requires_grad=True)
    y = torch.tensor([[0.2, float("nan"), 0.3], [float("nan"), 0.4, float("inf")]])
    reference_x = x.detach().clone().requires_grad_()
    index = TASKS.index(task)
    mean, std = module.target_mean[index], module.target_std[index]
    selected_x, selected_y = reference_x[valid], y[valid]
    denominator = max(1, selected_x.numel())
    expected_nll = F.gaussian_nll_loss(
        (selected_x - mean) / std, (selected_y - mean) / std,
        legacy.variance(task).expand_as(selected_x), full=True, reduction="sum",
    ) / denominator
    expected_aux = ((selected_x - selected_y) / std).square().sum() / denominator
    actual_nll, count = module(task, x, y, valid)
    actual_aux, aux_count = masked_standardized_mse(x, y, valid, std)
    torch.testing.assert_close(actual_nll, expected_nll)
    torch.testing.assert_close(actual_aux, expected_aux)
    assert count == aux_count == valid.sum()
    (expected_nll + expected_aux).backward()
    (actual_nll + actual_aux).backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(module.log_variances[task].grad, legacy.log_variances[task].grad)
    assert torch.isfinite(x.grad).all() and (x.grad[~valid] == 0).all()


def supervision_samples():
    support = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    support[0, :, -1, -1] = False
    support[1] = False
    weights = torch.zeros(2, 3, 4, 4)
    weights[:, 0, :2, :2] = 0.25
    weights[:, 1, 2:, 2:] = 0.25  # Partial support; third native cell has no footprint.
    return dict(
        prediction_valid=support, water=torch.full((2, 1, 4, 4), 0.3),
        water_valid=support.clone(), soil_moisture=torch.full((2, 1, 1, 1), 0.2),
        soil_moisture_valid=torch.ones(2, 1, 1, 1, dtype=torch.bool),
        et_native_weights=weights, et_native_value=torch.ones(2, 3),
        et_native_valid=torch.ones(2, 3, dtype=torch.bool),
        site_value=torch.tensor([[0.4, 0.25, 1.2]]).repeat(2, 1),
        site_valid=torch.ones(2, 3, dtype=torch.bool), site_footprint=weights[:, :1].clone(),
    )


@pytest.mark.parametrize("with_sites", [False, True])
@pytest.mark.parametrize("with_support", [False, True])
def test_shared_supervision_matches_original_pooling_and_gradients(with_sites, with_support):
    samples = supervision_samples()
    if not with_sites:
        samples = {k: v for k, v in samples.items() if not k.startswith("site_")}
    if not with_support:
        del samples["prediction_valid"]
    outputs = {t: torch.randn(2, 1, 4, 4, requires_grad=True) for t in TASKS}
    reference = {t: v.detach().clone().requires_grad_() for t, v in outputs.items()}
    plans = prepare_supervision(outputs, samples)
    for task in TASKS:
        pred, target, valid = combined_supervision_values(task, reference[task], samples)
        plan = plans[task]
        actual = plan.project(outputs[task])
        torch.testing.assert_close(actual, pred, rtol=0, atol=0)
        torch.testing.assert_close(plan.target, target, rtol=0, atol=0)
        torch.testing.assert_close(plan.valid, valid, rtol=0, atol=0)
        assert plan.count == valid.sum()
        actual.masked_fill(~valid, 0).square().sum().backward()
        pred.masked_fill(~valid, 0).square().sum().backward()
        torch.testing.assert_close(outputs[task].grad, reference[task].grad, rtol=0, atol=0)


class LossProbe(nn.Module):
    _loss_from_outputs = HydroStateModel._loss_from_outputs

    def __init__(self, statistics):
        super().__init__()
        self.loss_module = UncertaintyWeightedNLL(statistics)


def test_combined_head_losses_have_no_dynamic_selection_or_scalar_reads(target_statistics):
    model = LossProbe(target_statistics)
    samples = supervision_samples()
    outputs = {t: torch.randn(2, 1, 4, 4, requires_grad=True) for t in TASKS}
    auxiliary = {t: torch.randn_like(v, requires_grad=True) for t, v in outputs.items()}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        losses = model._loss_from_outputs(outputs, samples, auxiliary, dict.fromkeys(TASKS, 0.2))
        sum(v for k, v in losses.items() if k.startswith("loss_")).backward()
    operations = {event.key for event in profile.key_averages()}
    assert not operations & {"aten::masked_select", "aten::nonzero", "aten::_local_scalar_dense"}
    for task in TASKS:
        prediction, target, valid = combined_supervision_values(task, outputs[task], samples)
        expected, count = model.loss_module(task, prediction, target, valid)
        torch.testing.assert_close(losses[f"loss_{task}"], expected)
        assert losses[f"valid_count_{task}"] == count


class DistributedMultiHeadProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.slopes = nn.Parameter(torch.tensor([0.3, 0.4, 0.5, 0.6]))

    def forward(self, x, masks):
        sums = [((slope * x).square().masked_fill(~masks[i % 2], 0)).sum()
                for i, slope in enumerate(self.slopes)]
        return global_observation_means(
            sums, masks.sum(dim=1), torch.tensor([0, 1, 0, 1]),
        )


def distributed_merged_worker(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(DistributedMultiHeadProbe())
        for empty in (False, True):
            all_x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
            all_masks = torch.tensor([
                [[1, 0, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 1, 1, 0]],
            ], dtype=torch.bool)
            if empty:
                all_masks.zero_()
            model.zero_grad(set_to_none=True)
            actual, counts = model(all_x[rank], all_masks[rank])
            sum(actual).backward()
            slopes = model.module.slopes.detach().clone().requires_grad_()
            expected = [
                (slopes[i] * all_x[all_masks[:, i % 2]]).square().sum()
                / max(1, int(all_masks[:, i % 2].sum()))
                for i in range(4)
            ]
            sum(expected).backward()
            torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
            torch.testing.assert_close(counts, all_masks.sum(dim=(0, 2)).float())
            torch.testing.assert_close(model.module.slopes.grad, slopes.grad)
    finally:
        dist.destroy_process_group()


def test_merged_distributed_reduction_preserves_unequal_and_empty_support(tmp_path):
    mp.spawn(distributed_merged_worker, args=(str(tmp_path / "ddp"),), nprocs=2, join=True)


def test_causal_mse_keeps_objective_in_combined_head_path(target_statistics):
    from hydrostate.experiments.sm_causal import CausalNLL

    model = LossProbe(target_statistics)
    model.loss_module = CausalNLL(objective="mse", target_statistics=target_statistics)
    outputs = {t: torch.randn(2, 1, 4, 4, requires_grad=True) for t in TASKS}
    samples = supervision_samples()
    actual = model._loss_from_outputs(outputs, samples)
    for task in TASKS:
        prediction, target, valid = combined_supervision_values(task, outputs[task], samples)
        expected, _ = model.loss_module(task, prediction, target, valid)
        torch.testing.assert_close(actual[f"loss_{task}"], expected)


@pytest.mark.parametrize("condition", ["none", "relative", "zero"])
def test_causal_grouping_handles_prepared_cpu_metadata(condition):
    from hydrostate.experiments.sm_causal import CausalOlmoEarthEncoder

    encoder = CausalOlmoEarthEncoder.__new__(CausalOlmoEarthEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = CaptureBackbone()
    encoder.backbone.timestamp_error_tolerance = timedelta(days=15)
    encoder.out_channels = 1
    encoder.time_condition = condition
    if condition != "none":
        encoder.time_adapter = nn.Linear(24, 1)
    inputs = temporal_inputs()
    # Conditional features read all timestamps, including unavailable slots.
    inputs["observation_time"][~inputs["sensor_valid"]] = 0
    inputs["sm_target_time"] = torch.tensor([1600000000, 1600000000])
    expected = encoder(inputs)
    inputs["_temporal_metadata"], inputs["_temporal_indices"] = prepare_temporal_metadata(
        inputs["sensor_valid"], inputs["observation_time"]
    )
    torch.testing.assert_close(encoder(inputs), expected, rtol=0, atol=0)
