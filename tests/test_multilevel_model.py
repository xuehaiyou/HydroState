"""Multilevel extraction against actual OlmoEarth blocks, not a fake transformer."""

import copy
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope
from torch import nn

from hydrostate.models.multilevel_encoder import MultiLevelOlmoEarthEncoder, pool_modalities
from hydrostate.models.multilevel_model import OutputTaskAdapter
from hydrostate.normalization import LEGACY_TASK_NAMES as TASKS
from hydrostate.registry import MODELS


@MODELS.register_module(force=True)
class MultiLevelTestEncoder(nn.Module):
    def forward(self, inputs):
        return inputs


def tiny_encoder(monkeypatch, tmp_path, checkpoint=False, registers=0):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    config = EncoderConfig(
        supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
        embedding_size=64, num_heads=4, depth=3, drop_path=0,
        num_register_tokens=registers, use_flash_attn=False,
        use_linear_patch_embed=True, position_encoding="rope_3d_mixed",
    )
    core = config.build()
    monkeypatch.setattr(wrapper, "load_model_from_path", lambda *a, **kw:
                        SimpleNamespace(encoder=core))
    return MultiLevelOlmoEarthEncoder(
        model_path=str(tmp_path), embedding_channels=64, out_indices=(0, 1, 2),
        gradient_checkpointing=checkpoint,
    )


def raw_inputs():
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 1] = False
    times = torch.tensor([1600000000, 1602678400])[None, :, None].expand(2, 2, 3)
    inputs = dict(sensor_valid=valid, observation_time=times)
    for key, channels in (("s1", 2), ("s2", 12), ("landsat", 11)):
        inputs[key] = torch.randn(2, 2, channels, 12, 12)
        inputs[f"{key}_valid"] = torch.ones(2, 2, 1, 12, 12, dtype=torch.bool)
    return inputs


@pytest.mark.parametrize("checkpoint", [False, True])
def test_actual_encoder_depths_masks_gradients_and_repeated_forward(
    monkeypatch, tmp_path, checkpoint,
):
    encoder = tiny_encoder(monkeypatch, tmp_path, checkpoint).train()
    inputs = raw_inputs()
    actual = encoder(inputs)
    assert [x.shape for x in actual] == [(2, 64, 3, 3)] * 3
    sample, _, _ = encoder.backbone._prepare_modality_inputs(encoder._make_context(inputs))
    official = encoder.backbone.model(sample, patch_size=4, fast_pass=False)["tokens_and_masks"]
    values = {name: getattr(official, name) for name in
              ("sentinel1", "sentinel2_l2a", "landsat")}
    masks = {f"{name}_mask": getattr(official, f"{name}_mask") for name in values}
    torch.testing.assert_close(actual[-1], pool_modalities(values, masks), atol=1e-6, rtol=1e-5)
    assert not torch.allclose(actual[0], actual[-1])
    for _ in range(2):
        encoder.zero_grad(set_to_none=True)
        features = encoder(inputs)
        sum(x[:, 0].square().mean() for x in features).backward()
        core = encoder.backbone.model
        assert all(next(block.parameters()).grad is not None for block in core.blocks)
        gradients = [p.grad for p in core.patch_embeddings.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
        assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
        assert all(not p.requires_grad for p in core.project_and_aggregate.parameters())


def test_eval_is_invariant_to_missing_modalities_and_padded_timesteps(monkeypatch, tmp_path):
    encoder = tiny_encoder(monkeypatch, tmp_path).eval()
    inputs = raw_inputs()
    inputs["sensor_valid"][0, :, 1:] = False  # First sample is S1 only, one timestep.
    alone = {key: value[:1] for key, value in inputs.items()}
    with torch.no_grad():
        batch, single = encoder(inputs), encoder(alone)
    for actual, expected in zip(batch, single):
        torch.testing.assert_close(actual[:1], expected, atol=2e-6, rtol=1e-5)


def test_adapter_starts_as_identity_but_learns_a_task_adjustment():
    adapter = OutputTaskAdapter()
    assert sum(p.numel() for p in adapter.parameters()) == 4448
    x = torch.randn(2, 64, 4, 4)
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    adapter(x).square().mean().backward()
    optimizer.step()
    assert not torch.equal(adapter(x), x)
    adapter.zero_grad()
    adapter(x).square().mean().backward()
    assert adapter.layers[0].weight.grad.abs().sum() > 0


def make_model(config_name, statistics, small=False):
    cfg = Config.fromfile(f"configs/experiments/legacy/{config_name}.py")
    cfg.model.encoder = dict(type="MultiLevelTestEncoder")
    cfg.model.target_statistics = statistics
    if small:
        cfg.model.decoder.update(in_channels=8, channels=[8, 8, 4])
        cfg.model.adapter.bottleneck_channels = 2
        cfg.model.interaction.channels = 4
        for head in cfg.model.heads.values():
            head.update(in_channels=4, hidden_channels=2)
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        return MODELS.build(cfg.model)


@pytest.mark.parametrize("config_name", [
    "hydrostate_multilevel", "hydrostate_multilevel_independent", "hydrostate_multilevel_shared",
])
def test_configs_full_grid_predict_loss_and_checkpoint(config_name, target_statistics):
    model = make_model(config_name, target_statistics).eval()
    with torch.no_grad():
        outputs = model(tuple(torch.randn(1, 768, 30, 30) for _ in range(3)))
    assert set(outputs) == set(TASKS)
    assert all(x.shape == (1, 1, 40, 40) and torch.isfinite(x).all() for x in outputs.values())
    assert ((outputs["water"] >= 0) & (outputs["water"] <= 1)).all()
    assert (outputs["et"] >= 0).all()
    model = make_model(config_name, target_statistics, small=True)
    features = tuple(torch.randn(2, 8, 3, 3, requires_grad=True) for _ in range(3))
    data = dict(
        water=torch.full((2, 1, 4, 4), 0.2), water_valid=torch.ones(2, 1, 4, 4).bool(),
        soil_moisture=torch.full((2, 1, 1, 1), 0.3),
        soil_moisture_valid=torch.ones(2, 1, 1, 1).bool(),
        et_native_weights=torch.full((2, 1, 4, 4), 1 / 16),
        et_native_value=torch.ones(2, 1), et_native_valid=torch.ones(2, 1).bool(),
    )
    losses = model(features, data, mode="loss")
    assert not any("aux" in name for name in losses)
    sum(v for k, v in losses.items() if k.startswith("loss_")).backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in features)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    predicted = model(features, data, mode="predict")
    assert "soil_moisture_prediction_valid" in predicted[0]
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    for name, value in model(features).items():
        torch.testing.assert_close(restored(features)[name], value)


def test_learnable_interaction_strength_is_task_specific_and_disabled_has_no_parameters():
    from hydrostate.models.task_branches import TaskGatedInteraction

    layer = TaskGatedInteraction(channels=4, learnable_scale=True)
    values = {task: torch.randn(2, 4, 3, 3) for task in TASKS}
    output = layer(values)
    output["water"].square().mean().backward()
    assert layer.task_scales["water"].grad is not None
    assert layer.task_scales["soil_moisture"].grad is None
    disabled = TaskGatedInteraction(channels=4, mode="none", learnable_scale=True)
    assert not list(disabled.parameters())
    assert all(disabled(values)[task] is values[task] for task in TASKS)
