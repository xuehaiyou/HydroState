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
    ],
)
def test_layer_id(name, expected):
    assert olmoearth_layer_id(name, 12) == expected


def test_layer_scales_are_monotonic():
    rate = 0.65
    scales = [rate ** (13 - layer) for layer in range(14)]
    assert all(left < right for left, right in zip(scales, scales[1:]))


def test_unused_torch_import_is_available():
    assert torch.__version__

