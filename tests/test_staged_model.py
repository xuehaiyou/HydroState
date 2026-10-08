"""Architecture contracts and loss/gradient semantics for staged task retrieval."""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmengine.config import Config
from mmengine.registry import DefaultScope
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from hydrostate.engine.layer_decay_constructor import OlmoEarthLayerDecayOptimWrapperConstructor
from hydrostate.losses.uncertainty_nll import masked_standardized_mse
from hydrostate.models import SharedUNetDecoder, StagedHydroStateModel, TaskGatedInteraction
from hydrostate.models.hydrostate_model import combined_supervision_values
from hydrostate.normalization import TASKS
from hydrostate.registry import MODELS


@MODELS.register_module(force=True)
class StagedTestEncoder(nn.Module):
    def forward(self, inputs):
        return inputs


def make_model(statistics, **overrides):
    config = dict(
        encoder=dict(type="StagedTestEncoder"),
        decoder=dict(type="StagedTaskDecoder", in_channels=8, channels=[8, 4, 4]),
        heads={task: dict(type="RegressionHead", in_channels=4,
                         activation="softplus" if task == "et" else "sigmoid")
               for task in TASKS},
        loss=dict(type="UncertaintyWeightedNLL"),
        target_statistics=statistics,
    )
    config.update(overrides)
    return StagedHydroStateModel(**config)


def samples(batch=2):
    support = torch.ones(batch, 1, 4, 4, dtype=torch.bool)
    support[..., -1, -1] = False
    weights = torch.zeros(batch, 2, 4, 4)
    weights[:, 0, :2, :2] = 0.25
    weights[:, 1, 2:, 2:] = 0.25  # Incomplete input support: must be excluded.
    return dict(
        prediction_valid=support,
        water=torch.full((batch, 1, 4, 4), 0.3), water_valid=support.clone(),
        soil_moisture=torch.full((batch, 1, 1, 1), 0.2),
        soil_moisture_valid=torch.ones(batch, 1, 1, 1, dtype=torch.bool),
        et_native_weights=weights, et_native_value=torch.ones(batch, 2),
        et_native_valid=torch.ones(batch, 2, dtype=torch.bool),
        site_value=torch.tensor([[0.4, 0.25, 1.2]]).repeat(batch, 1),
        site_valid=torch.ones(batch, 3, dtype=torch.bool),
        site_footprint=weights[:, :1].clone(),
    )


def test_shared_stages_preserve_baseline_output_and_checkpoint():
    decoder = SharedUNetDecoder(in_channels=8, channels=[8, 4, 4])
    x = torch.randn(2, 8, 3, 3)
    stages = decoder.forward_stages(x)
    assert [v.shape for v in stages] == [(2, 8, 3, 3), (2, 4, 6, 6), (2, 4, 12, 12)]
    torch.testing.assert_close(decoder(x), stages[-1], rtol=0, atol=0)
    assert all(k.startswith(("stem.", "blocks.")) for k in decoder.state_dict())
    restored = copy.deepcopy(decoder)
    restored.load_state_dict(decoder.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), decoder(x), rtol=0, atol=0)


@pytest.mark.parametrize("config_name,channels", [
    ("hydrostate_staged", [512, 256, 128]),
    ("hydrostate_staged_independent", [512, 256, 128]),
    ("hydrostate_staged_softmax", [512, 256, 128]),
    ("hydrostate_staged_light", [256, 128, 64]),
])
def test_real_config_build_and_full_resolution(config_name, channels, target_statistics):
    cfg = Config.fromfile(f"configs/{config_name}.py")
    assert cfg.model.decoder.channels == channels
    assert cfg.model.encoder.embedding_channels == 768
    cfg.model.encoder = dict(type="StagedTestEncoder")
    cfg.model.target_statistics = target_statistics
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        model = MODELS.build(cfg.model).eval()
    assert model.decoder.out_channels == channels[-1]
    for heads in (model.heads, model.auxiliary_heads):
        assert all(head.layers[0].in_channels == channels[-1] for head in heads.values())
    assert all(layer.in_channels == channels[-1] for layer in model.interaction.messages.values())
    with torch.no_grad():
        outputs = model(torch.randn(1, 768, 30, 30))
    assert set(outputs) == set(TASKS)
    assert all(v.shape == (1, 1, 40, 40) and torch.isfinite(v).all() for v in outputs.values())
    assert ((outputs["water"] > 0) & (outputs["water"] < 1)).all()
    assert ((outputs["soil_moisture"] > 0) & (outputs["soil_moisture"] < 1)).all()
    assert (outputs["et"] > 0).all()


@pytest.mark.parametrize("attention,adapter,fusion", [
    ("none", "none", "add"), ("se", "bottleneck", "concat"),
    ("cbam", "depthwise", "add"),
])
def test_branch_variants_backpropagate(target_statistics, attention, adapter, fusion):
    model = make_model(target_statistics, decoder=dict(
        type="StagedTaskDecoder", in_channels=8, channels=[8, 4, 4], fusion=fusion,
        attention=dict(kind=attention), adapter=dict(kind=adapter),
    ))
    losses = model(torch.randn(2, 8, 3, 3), samples(), mode="loss")
    total = sum(v for k, v in losses.items() if k.startswith("loss_"))
    total.backward()
    assert torch.isfinite(total)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    restored = make_model(target_statistics, decoder=dict(
        type="StagedTaskDecoder", in_channels=8, channels=[8, 4, 4], fusion=fusion,
        attention=dict(kind=attention), adapter=dict(kind=adapter),
    ))
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 8, 3, 3)
    for task in TASKS:
        torch.testing.assert_close(restored(x)[task], model(x)[task])


@pytest.mark.parametrize("mode,coefficient", [("sigmoid", 0.25), ("softmax", 0.5)])
def test_interaction_is_simultaneous_and_matches_formula(mode, coefficient):
    layer = TaskGatedInteraction(channels=1, mode=mode, residual_scale=0.2)
    for projection in layer.messages.values():
        nn.init.ones_(projection.weight)
    for gate in layer.gates.values():
        nn.init.zeros_(gate.weight)
        nn.init.zeros_(gate.bias)
    features = {task: torch.full((1, 1, 2, 2), v, requires_grad=True)
                for task, v in zip(TASKS, (1.0, 2.0, 4.0))}
    outputs = layer(features)
    reordered = layer(dict(reversed(list(features.items()))))
    for task in TASKS:
        expected = features[task] + 0.2 * coefficient * sum(
            value for source, value in features.items() if source != task
        )
        torch.testing.assert_close(outputs[task], expected)
        torch.testing.assert_close(outputs[task], reordered[task])
    outputs["water"].sum().backward()
    for task in ("soil_moisture", "et"):
        torch.testing.assert_close(features[task].grad, torch.full_like(features[task],
                                                                      0.2 * coefficient))


def test_disabled_interaction_is_identity_and_has_no_parameters():
    layer = TaskGatedInteraction(mode="none")
    features = {task: torch.randn(2, 4, 4, 4) for task in TASKS}
    assert not list(layer.parameters())
    assert all(layer(features)[task] is features[task] for task in TASKS)


def test_auxiliary_support_objective_and_gradient_routes(target_statistics):
    model = make_model(target_statistics, auxiliary_weight=0.3)
    x, data = torch.randn(2, 8, 3, 3), samples()
    final, auxiliary = model._forward_with_auxiliary(x, auxiliary=True)
    losses = model(x, data, mode="loss")
    for i, task in enumerate(TASKS):
        pred, target, valid = combined_supervision_values(task, auxiliary[task], data)
        residual = (pred[valid] - target[valid]) / model.loss_module.target_std[i]
        expected = 0.3 * residual.square().mean()
        torch.testing.assert_close(losses[f"loss_aux_{task}"], expected)
    assert losses["valid_count_water"] == 32  # 15 pixels + one site, each sample.
    assert losses["valid_count_soil_moisture"] == 4
    assert losses["valid_count_et"] == 4  # One complete PML cell + one site per sample.
    losses["loss_aux_soil_moisture"].backward()
    assert any(p.grad is not None for p in model.decoder.branches["soil_moisture"].parameters())
    assert all(p.grad is None for p in model.interaction.parameters())
    assert all(p.grad is None for p in model.heads.parameters())
    assert all(p.grad is None for p in model.loss_module.parameters())
    model.zero_grad(set_to_none=True)
    model(x, data, mode="loss")["loss_soil_moisture"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.decoder.branches["water"].parameters())
    assert all(p.grad is None for p in model.auxiliary_heads.parameters())


def test_auxiliary_heads_only_execute_in_loss_mode(target_statistics):
    model = make_model(target_statistics).eval()
    calls = []
    handles = [head.register_forward_hook(lambda *args: calls.append(1))
               for head in model.auxiliary_heads.values()]
    x = torch.randn(2, 8, 3, 3)
    model(x)
    predictions = model(x, samples(), mode="predict")
    assert len(predictions) == 2 and not calls
    assert all(task in predictions[0] for task in TASKS)
    model(x, samples(), mode="loss")
    assert len(calls) == 3
    for handle in handles:
        handle.remove()
    disabled = make_model(target_statistics, auxiliary_weight=0)
    assert not list(disabled.auxiliary_heads.parameters())
    assert not any(k.startswith("loss_aux_") for k in disabled(x, samples(), mode="loss"))


def test_config_options_select_math_and_per_task_auxiliary_weights(target_statistics):
    cfg = Config.fromfile("configs/hydrostate_staged.py")
    cfg.merge_from_dict({
        "model.decoder.attention.kind": "se",
        "model.decoder.adapter.kind": "depthwise",
        "model.decoder.fusion": "concat",
        "model.interaction.mode": "softmax",
        "model.auxiliary_weight.water": 0.0,
    })
    cfg.model.encoder = dict(type="StagedTestEncoder")
    cfg.model.decoder.in_channels = 8
    cfg.model.decoder.channels = [8, 4, 4]
    cfg.model.target_statistics = target_statistics
    for head in cfg.model.heads.values():
        head.in_channels = 4
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        model = MODELS.build(cfg.model)
    losses = model(torch.randn(2, 8, 3, 3), samples(), mode="loss")
    assert set(model.auxiliary_heads) == {"soil_moisture", "et"}
    assert "loss_aux_water" not in losses and "loss_aux_soil_moisture" in losses
    assert model.interaction.mode == "softmax"
    cfg.merge_from_dict({"model.auxiliary_weight": 0.0})
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        assert not len(MODELS.build(cfg.model).auxiliary_heads)


def test_all_missing_labels_produce_zero_finite_gradients(target_statistics):
    model = make_model(target_statistics)
    data = samples()
    for key, value in data.items():
        if key.endswith("_valid") and key != "prediction_valid":
            value.fill_(False)
    for key in ("water", "soil_moisture", "et_native_value", "site_value"):
        data[key].fill_(float("nan"))
    losses = model(torch.randn(2, 8, 3, 3), data, mode="loss")
    total = sum(v for k, v in losses.items() if k.startswith("loss_"))
    assert total == 0
    total.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and (parameter.grad == 0).all()


def test_new_modules_receive_full_learning_rate(target_statistics):
    model = make_model(target_statistics)
    constructor = OlmoEarthLayerDecayOptimWrapperConstructor(
        dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05)),
        dict(num_layers=12, layer_decay_rate=0.65),
    )
    groups = []
    constructor.add_params(groups, model)
    actual = {id(p): g["lr"] for g in groups for p in g["params"]}
    for module in (model.decoder.branches, model.interaction, model.auxiliary_heads):
        assert all(actual[id(p)] == 1e-4 for p in module.parameters())


@pytest.mark.parametrize("kwargs", [
    dict(auxiliary_weight=-1), dict(auxiliary_weight={"water": 0.2}),
    dict(interaction=dict(mode="bad")), dict(interaction=dict(channels=99)),
    dict(interaction=dict(residual_scale=float("nan"))),
    dict(decoder=dict(type="StagedTaskDecoder", in_channels=8, channels=[8, 4, 4],
                      attention=dict(kind="bad"))),
    dict(decoder=dict(type="StagedTaskDecoder", in_channels=8, channels=[8, 4, 4],
                      adapter=dict(reduction=0))),
])
def test_invalid_configuration_fails_early(target_statistics, kwargs):
    with pytest.raises(ValueError):
        make_model(target_statistics, **kwargs)


class AuxiliaryDDPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, valid):
        return masked_standardized_mse(self.slope * x, torch.zeros_like(x), valid, 0.2)


def auxiliary_ddp_worker(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(AuxiliaryDDPProbe())
        for counts in ((2, 3), (0, 3), (0, 0)):
            x = torch.arange(1, counts[rank] + 1, dtype=torch.float32)
            model.zero_grad(set_to_none=True)
            value, count = model(x, torch.ones_like(x, dtype=torch.bool))
            value.backward()
            whole = torch.cat([torch.arange(1, n + 1, dtype=torch.float32) for n in counts])
            slope = torch.tensor(0.5, requires_grad=True)
            expected = (slope * whole / 0.2).square().sum() / max(1, sum(counts))
            expected.backward()
            torch.testing.assert_close(value, expected)
            torch.testing.assert_close(model.module.slope.grad, slope.grad)
            assert count == sum(counts)
    finally:
        dist.destroy_process_group()


def test_auxiliary_ddp_matches_global_reference(tmp_path):
    mp.spawn(auxiliary_ddp_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
