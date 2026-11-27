import torch

from hydrostate.models.cosa.landsat_terrafm import LandsatTerraBDTADecoder, align_level, band_order


def test_native_grid_coordinates_and_missing_support():
    # Native centers at 2 and 6 on the horizontal axis match the two output centers.
    native = torch.tensor([[[[3.0, 9.0]]]])
    supported = torch.ones_like(native, dtype=torch.bool)
    torch.testing.assert_close(align_level(native, supported, supported, 4, 2), native)
    mask = torch.tensor([[[[True, False]]]])
    expected = torch.tensor([[[[3.0, 0.0]]]])
    torch.testing.assert_close(align_level(native, mask, supported, 4, 2), expected)
    torch.testing.assert_close(align_level(native, supported, mask, 4, 2), expected)


def test_landsat_band_order_and_resnet_stage_projections():
    assert band_order("landsat", [f"B{i}" for i in range(1, 12)]) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        8,
        9,
        10,
    ]
    model = LandsatTerraBDTADecoder(in_channels=768, channels=32, num_levels=5, bypass_heads=2)
    assert [p[1].in_features for p in model.projections] == [768] * 5
    assert [p[1].in_features for p in model.anysat_projections] == [64, 256, 512, 1024, 2048]
    assert model.learn_source_weights and model.source_names == ("terrafm", "landsat")
