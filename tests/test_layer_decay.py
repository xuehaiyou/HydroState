import pytest
import torch

from hydrostate.engine.layer_decay_constructor import olmoearth_layer_id


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("encoder.backbone.model.patch_embed.weight", 0),
        ("encoder.backbone.model.blocks.0.attn.weight", 1),
        ("encoder.backbone.model.blocks.11.attn.weight", 12),
        ("decoder.stem.0.weight", 13),
        ("heads.water.layers.0.weight", 13),
        ("auxiliary_heads.water.layers.0.weight", 13),
        ("interaction.messages.water_to_et.weight", 13),
        ("loss_module.log_variances.water", 13),
    ],
)
def test_layer_id(name, expected):
    assert olmoearth_layer_id(name, 12) == expected


def test_layer_scales_are_monotonic():
    rate = 0.65
    scales = [rate ** (13 - layer) for layer in range(14)]
    assert all(left < right for left, right in zip(scales, scales[1:]))


def test_scalar_uncertainty_has_full_learning_rate_and_no_weight_decay(target_statistics):
    from hydrostate.engine.layer_decay_constructor import OlmoEarthLayerDecayOptimWrapperConstructor
    from hydrostate.losses import UncertaintyWeightedNLL

    model = torch.nn.Module()
    model.loss_module = UncertaintyWeightedNLL(target_statistics)
    constructor = OlmoEarthLayerDecayOptimWrapperConstructor(
        dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05)),
        dict(num_layers=12, layer_decay_rate=0.65),
    )
    groups = []
    constructor.add_params(groups, model)
    assert len(groups) == 1 and len(groups[0]["params"]) == 3
    assert groups[0]["lr"] == 1e-4 and groups[0]["weight_decay"] == 0


def test_official_recipe_matches_upstream_parameter_depths_and_learning_rates():
    from rslearn.models.olmoearth_pretrain.optimizer import get_layer_id

    from hydrostate.engine.layer_decay_constructor import (
        OlmoEarthLayerDecayOptimWrapperConstructor,
        official_olmoearth_layer_id,
    )

    names = ["patch_embeddings.proj.weight", "composite_encodings.month_embed.weight",
             "blocks.0.attn.qkv.weight", "blocks.11.mlp.fc1.bias", "norm.weight",
             "project_and_aggregate.projection.weight"]
    for name in names:
        ours = official_olmoearth_layer_id(f"encoder.backbone.model.{name}", 12)
        upstream = get_layer_id(f"model.encoder.0.model.{name}", 12)
        assert ours == upstream
    for name in ("decoder.laterals.0.weight", "adapters.water.layers.0.weight",
                 "interaction.task_scales.water", "heads.water.layers.0.weight"):
        assert official_olmoearth_layer_id(name, 12) == 12
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.backbone = torch.nn.Module()
    core = model.encoder.backbone.model = torch.nn.Module()
    core.patch_embeddings = torch.nn.Linear(2, 2)
    core.norm = torch.nn.LayerNorm(2)
    core.blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(12))
    constructor = OlmoEarthLayerDecayOptimWrapperConstructor(
        dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05)),
        dict(num_layers=12, layer_decay_rate=0.65, recipe="official"),
    )
    groups = []
    constructor.add_params(groups, model)
    rates = {id(p): group["lr"] for group in groups for p in group["params"]}
    assert rates[id(core.patch_embeddings.weight)] == pytest.approx(1e-4 * 0.65**12)
    assert rates[id(core.blocks[0].weight)] == rates[id(core.patch_embeddings.weight)]
    assert rates[id(core.blocks[-1].weight)] == pytest.approx(6.5e-5)
    assert rates[id(core.norm.weight)] == 1e-4
    assert all(group["weight_decay"] == 0.05 for group in groups)


def test_multilevel_cosine_schedule_preserves_layer_ratios():
    from mmengine.config import Config
    from mmengine.optim.scheduler import CosineAnnealingLR

    cfg = Config.fromfile("configs/hydrostate_multilevel.py")
    optimizer = torch.optim.AdamW([
        dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=1e-4 * 0.65**12),
        dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=1e-4),
    ])
    options = dict(cfg.param_scheduler[1])
    options.pop("type")
    scheduler = CosineAnnealingLR(optimizer, **options)
    for _ in range(50):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"] == pytest.approx(
            0.65**12
        )
