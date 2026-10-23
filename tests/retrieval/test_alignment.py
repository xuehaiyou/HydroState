"""Native-band storage must retain spectral order, location and date support."""

import numpy as np

from hydrostate.data.gee.products import BANDS
from hydrostate.data.retrieval.alignment import align_inputs, grid_metadata, nearest_grid
from hydrostate.data.retrieval.composites import normalize
from hydrostate.data.retrieval.contract import INPUT_GROUPS, RetrievalSchema


def test_noninteger_pixel_centres_masks_and_dates():
    # 15 m source pixel centres: 7.5, 22.5, 37.5, 52.5; target: 5,15,25,...
    source = np.arange(8).reshape(2, 4)
    expected = source[[0, 1, 1]][:, [0, 1, 1, 2, 3, 3]]
    np.testing.assert_array_equal(nearest_grid(source, (3, 6)), expected)
    np.testing.assert_array_equal(nearest_grid(source == 1, (3, 6)), expected == 1)
    assert nearest_grid(source, source.shape) is source


def test_band_order_missing_group_and_independent_quality():
    schema = RetrievalSchema()
    inputs = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.inputs.items()}
    for index, (name, group) in enumerate(INPUT_GROUPS.items()):
        for channel, band in enumerate(group.bands):
            inputs[name][:, channel] = BANDS[group.sensor].index(band) + 1
        inputs[f"{name}_valid"][:] = 1
        inputs[f"{name}_quality"][:, 0] = index + 1
        inputs[f"{name}_quality"][:, 1:] = index
    # Coarse-band missingness must mask the corresponding footprint in ALL S2 channels.
    inputs["s2_60m_valid"][0, :, 0, 0] = 0
    inputs["s2_60m_quality"][0, :, 0, 0] = 0
    aligned = align_inputs(inputs)
    assert set(aligned) == set(schema.model_inputs)
    for name, spec in schema.model_inputs.items():
        assert aligned[name].shape == spec.tail_shape
        assert aligned[name].dtype == np.dtype(spec.dtype)
    for sensor, bands in BANDS.items():
        np.testing.assert_array_equal(aligned[sensor][1, :, 0, 0], np.arange(1, len(bands) + 1))
    assert not aligned["s2_valid"][0, :, :6, :6].any()
    assert not aligned["s2"][0, :, :6, :6].any()
    assert aligned["s2_valid"][0, 0, 6, 6]
    assert aligned["s2_10m_quality"][0, 0, 6, 6] == 2
    assert aligned["s2_60m_quality"][0, 0, 6, 6] == 4


def test_normalization_is_band_named_not_group_position():
    values = np.array([[[0.2]], [[290]], [[306]]], dtype=np.float32)
    actual = normalize("landsat", values, np.ones((1, 1), bool), {}, ["B8", "B10", "B11"])
    np.testing.assert_array_equal(actual[:, 0, 0], np.array([0.6, 0.5, 0.7], np.float16))
    pan = normalize("landsat", values[:1], np.ones((1, 1), bool), {}, ["B8"])
    assert pan[0, 0, 0] == np.float16(0.6)
    assert not normalize("landsat", values, np.zeros((1, 1), bool), {}, ["B8", "B10", "B11"]).any()


def test_native_grids_have_the_same_outer_bounds():
    sample = dict(crs="EPSG:32650", transform=[10, 0, 400005, 0, -10, 3000015])
    for grid in grid_metadata(sample).values():
        a, b, left, d, e, top = grid["transform"]
        height, width = grid["shape"]
        assert (left, top, b, d) == (400005, 3000015, 0, 0)
        assert (a * width, -e * height) == (1200, 1200)
