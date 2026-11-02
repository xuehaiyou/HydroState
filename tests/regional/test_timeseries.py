import numpy as np

from hydrostate.inference.regional_timeseries import block_totals


def test_conditional_sm_and_total_et_have_different_area_aggregation():
    water = np.array([0.8, 0.2, 0, np.nan])
    land = np.array([0.2, 0.8, 1, np.nan])
    et = np.array([6, 2, 1, np.nan])
    sm = np.array([0.4, 0.1, 0.3, np.nan])
    masks = {t: np.ones(4, bool) for t in ('water', 'land', 'et_total', 'soil_moisture')}
    masks['soil_moisture'][2] = False
    totals = block_totals(water, land, et, sm, masks)
    np.testing.assert_allclose(totals, [1, 3, 9, 3, 0.16, 1, 2])


def test_valid_zero_sm_is_included_and_water_only_sm_is_excluded():
    masks = {t: np.ones(2, bool) for t in ('water', 'land', 'et_total', 'soil_moisture')}
    result = block_totals(np.array([0, 1]), np.array([1, 0]),
                          np.array([0, 0]), np.array([0, 0.9]), masks)
    np.testing.assert_allclose(result, [1, 2, 0, 2, 0, 1, 1])
