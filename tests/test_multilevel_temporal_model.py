"""Multilevel time queries preserve observation masks and independent output dates."""

import copy
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.models.multilevel_temporal_model import (
    MultiLevelTemporalHydroStateModel,
    MultiLevelTemporalOlmoEarthEncoder,
)
from hydrostate.models.task_branches import TaskGatedInteraction
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
from hydrostate.tasks import task_config


def install_tiny_backbone(monkeypatch):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    def load(*args, **kwargs):
        core = EncoderConfig(
            supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
            embedding_size=32, num_heads=4, depth=3, drop_path=0,
            num_register_tokens=0, use_flash_attn=False,
            use_linear_patch_embed=True, position_encoding="rope_3d_mixed",
        ).build()
        return SimpleNamespace(encoder=core)

    monkeypatch.setattr(wrapper, "load_model_from_path", load)


def raw_inputs(tasks=()):
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 1] = False
    times = torch.tensor([[1600000000, 1600003600, 1600007200],
                          [1600086400, 1600086400, 1600086400]])[None].repeat(2, 1, 1)
    inputs = dict(sensor_valid=valid, observation_time=times)
    for key, channels in (("s1", 2), ("s2", 12), ("landsat", 11)):
        inputs[key] = torch.randn(2, 2, channels, 12, 12)
        inputs[key + "_valid"] = torch.ones(2, 2, 1, 12, 12, dtype=torch.bool)
    if tasks:
        inputs["target_time"] = {task: torch.full((2, 2), 1600007200) for task in tasks}
    return inputs


def moments(tasks):
    return dict(version=2, split="train", ddof=0,
                support_version=CONFIGURED_SUPPORT_VERSION, task_specs=task_config(tasks),
                training_fingerprint="0" * 64,
                tasks={task: dict(count=3, mean=0, std=1) for task in tasks})


@pytest.mark.parametrize("checkpoint", [False, True])
def test_three_depths_in_one_pass_preserve_time_masks_and_gradients(
    monkeypatch, tmp_path, checkpoint,
):
    install_tiny_backbone(monkeypatch)
    encoder = MultiLevelTemporalOlmoEarthEncoder(
        model_path=str(tmp_path), embedding_channels=32, out_indices=(0, 1, 2),
        gradient_checkpointing=checkpoint,
    )
    inputs = raw_inputs()
    inputs["landsat_valid"][:, :, :, :4, :4] = False
    calls = []
    hooks = [block.register_forward_hook(lambda *args: calls.append(1))
             for block in encoder.backbone.model.blocks]
    levels = encoder(inputs)
    assert len(calls) == 3  # Three depths come from one traversal, not three encoders.
    for hook in hooks:
        hook.remove()
    assert len(levels) == 3
    for level in levels:
        assert level["features"].shape == (2, 6, 32, 3, 3)
        assert not level["valid"][0, [1, 3, 5]].any()
        assert not level["valid"][:, [4, 5], 0, 0].any()
        torch.testing.assert_close(level["time"], levels[-1]["time"])
        torch.testing.assert_close(level["valid"], levels[-1]["valid"])
    assert not torch.allclose(levels[0]["features"], levels[-1]["features"])
    sum(level["features"][:, :, 0].square().mean() for level in levels).backward()
    assert all(next(block.parameters()).grad is not None for block in encoder.backbone.model.blocks)
    encoder.eval()
    altered = copy.deepcopy(inputs)
    altered["landsat"][:, :, :, :4, :4] = 100000
    alone = {key: value[:1] for key, value in inputs.items()}
    with torch.no_grad():
        baseline, masked, single = encoder(inputs), encoder(altered), encoder(alone)
    for original, changed, one in zip(baseline, masked, single):
        torch.testing.assert_close(original["features"], changed["features"])
        torch.testing.assert_close(original["features"][:1], one["features"],
                                   atol=3e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["none", "sigmoid", "softmax"])
@pytest.mark.parametrize("merge_ops", [False, True])
def test_selected_interaction_matches_full_outputs_and_gradients(mode, merge_ops):
    layer = TaskGatedInteraction(channels=4, mode=mode, merge_ops=merge_ops,
                                learnable_scale=True)
    reference = copy.deepcopy(layer)
    values = {task: torch.randn(2, 4, 3, 3, requires_grad=True) for task in layer.tasks}
    other = {task: value.detach().clone().requires_grad_() for task, value in values.items()}
    selected = layer(values, targets=("water",))
    expected = reference(other)["water"]
    assert set(selected) == {"water"}
    torch.testing.assert_close(selected["water"], expected)
    selected["water"].square().mean().backward()
    expected.square().mean().backward()
    for task in values:
        if values[task].grad is None:
            assert other[task].grad is None
        else:
            torch.testing.assert_close(values[task].grad, other[task].grad)
    for (name, p), (_, q) in zip(layer.named_parameters(), reference.named_parameters()):
        if p.grad is None:
            # The full packed convolution materializes zero gradients on unused
            # recipient edges; a selected projection omits those edges entirely.
            assert q.grad is None or not q.grad.any(), name
        else:
            torch.testing.assert_close(p.grad, q.grad)
    for invalid in ((), ("water", "water"), ("unknown",)):
        with pytest.raises(ValueError, match="unique configured tasks"):
            layer(values, targets=invalid)


def test_multilevel_queries_adapters_loss_and_checkpoint(monkeypatch, tmp_path):
    install_tiny_backbone(monkeypatch)
    tasks = {name: dict(support="pixel", grid="output") for name in ("water", "land", "lst")}
    model = MultiLevelTemporalHydroStateModel(
        encoder=dict(type="MultiLevelTemporalOlmoEarthEncoder", model_path=str(tmp_path),
                     embedding_channels=32, out_indices=(0, 1, 2), gradient_checkpointing=True),
        decoder=dict(type="MultiLevelFusionDecoder", in_channels=32, channels=(8, 8, 4)),
        adapter=dict(bottleneck_channels=2),
        interaction=dict(channels=4, mode="sigmoid", learnable_scale=True),
        heads={name: dict(type="RegressionHead", in_channels=4, hidden_channels=4,
                         activation="identity") for name in tasks},
        tasks=tasks, fraction_groups=[["water", "land"]], target_statistics=moments(tasks),
        loss=dict(type="UncertaintyWeightedNLL"), query_time=dict(query_channels=8),
    ).eval()
    assert len({id(adapter) for adapter in model.adapters.values()}) == 3
    inputs = raw_inputs(tasks)
    baseline = model(inputs)
    altered = copy.deepcopy(inputs)
    altered["target_time"]["lst"] += 10 * 86400
    changed = model(altered)
    assert not torch.allclose(baseline["lst"], changed["lst"])
    # Interactions must use the recipient's date, not the source output's date.
    torch.testing.assert_close(baseline["water"], changed["water"], atol=0, rtol=0)
    torch.testing.assert_close(baseline["land"], changed["land"], atol=0, rtol=0)
    torch.testing.assert_close(baseline["water"] + baseline["land"],
                               torch.ones_like(baseline["water"]))
    targets = dict(prediction_valid=torch.ones(2, 1, 4, 4, dtype=torch.bool))
    for name in tasks:
        targets[name] = torch.rand(2, 1, 4, 4)
        targets[name + "_valid"] = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    model.train()
    losses = model(inputs, targets, mode="loss")
    sum(value for name, value in losses.items() if name.startswith("loss_")).backward()
    assert all(layer.weight.grad.abs().sum() > 0 for layer in model.decoder.laterals)
    assert all(adapter.layers[-1].weight.grad.abs().sum() > 0
               for adapter in model.adapters.values())
    assert model.query_time.relative_score[0].weight.grad.abs().sum() > 0
    assert model.interaction.messages["lst_to_water"].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    prediction = model(inputs, targets, mode="predict")[0]
    torch.testing.assert_close(prediction["target_time"]["lst"], inputs["target_time"]["lst"][0])
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for name, value in model(inputs).items():
            torch.testing.assert_close(restored(inputs)[name], value)
    altered["target_time"]["water"] += 1
    with pytest.raises(RuntimeError, match="share a target time"):
        model(altered)


def test_hpc_config_keeps_multilevel_architecture_and_all_sixteen_tasks():
    cfg = Config.fromfile("configs/hydrostate_ecosystem_multilevel_temporal_hpc.py")
    configure_data_contract(cfg)
    assert cfg.model.type == "MultiLevelTemporalHydroStateModel"
    assert cfg.model.encoder.out_indices == [3, 7, 11]
    assert cfg.model.decoder.type == "MultiLevelFusionDecoder"
    assert cfg.model.decoder.channels == [256, 128, 64]
    assert cfg.model.adapter.enabled and cfg.model.adapter.bottleneck_channels == 16
    assert cfg.model.interaction.mode == "sigmoid" and cfg.model.interaction.learnable_scale
    assert cfg.model.water_soil_coupling is None
    assert len(cfg.model.tasks) == len(cfg.model.heads) == 16
    assert all(head.in_channels == 64 and head.hidden_channels == 32
               for head in cfg.model.heads.values())
    assert cfg.target_time_queries and cfg.train_dataloader.dataset.target_time_queries
    assert cfg.label_packs[1].max_uncertainty.lst == 2
    assert cfg.optim_wrapper.paramwise_cfg.recipe == "official"
    assert cfg.param_scheduler[1].eta_min_ratio == 0.01
    assert cfg.train_dataloader.batch_size * cfg.optim_wrapper.accumulative_counts == 16
