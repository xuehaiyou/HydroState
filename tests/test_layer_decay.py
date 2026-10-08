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
