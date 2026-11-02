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


def test_summary_reuses_export_audit_and_recomputes_land_weighted_sm(tmp_path):
    import json

    import pandas as pd
    import zarr
    from shapely.geometry import box, mapping

    from hydrostate.inference.regional_timeseries import summarize

    root = tmp_path / 'region'
    product = root / 'products/test.zarr'
    product.parent.mkdir(parents=True)
    (root / 'plan.json').write_text(json.dumps(dict(
        identity='fixture', geometry=mapping(box(0, 0, 60, 60)), grid=dict(crs='EPSG:32650'),
    )))
    g = zarr.open_group(str(product), mode='w')
    g.attrs.update(identity=dict(plan='fixture'), transform=[30, 0, 0, 0, -30, 60])
    arrays = dict(water=[0.8, 0.2, 0, 0], land=[0.2, 0.8, 1, 1],
                  et_total=[6, 2, 1, 3], soil_moisture=[0.4, 0.1, 0.3, 0])
    for name, values in arrays.items():
        g.create_dataset(name, data=np.array(values, dtype='f4').reshape(1, 2, 2))
        g.create_dataset(name + '_valid', data=np.ones((1, 2, 2), dtype='u1'))
    g.create_dataset('date_complete', data=np.array([1], dtype='u1'))
    bounds = np.array([['2017-04-15', '2017-04-23']], dtype='datetime64[s]').astype('i8')
    g.create_dataset('time_bounds', data=bounds)
    (product.parent / 'test_2017-04-15.json').write_text(json.dumps(dict(
        target=dict(start='2017-04-15'), audit=dict(inside_pixels=4, variables=dict(
            water=dict(mean=0.25, valid_pixels=4), et_total=dict(mean=3, valid_pixels=4),
        )),
    )))
    summarize(product, tmp_path / 'report')
    row = pd.read_csv(tmp_path / 'report/timeseries.csv').iloc[0]
    np.testing.assert_allclose(row.water_area_km2, 0.0009)
    np.testing.assert_allclose(row.et_total_mean_mm_day, 3)
    np.testing.assert_allclose(row.land_sm_mean_m3_m3, 0.46 / 3, rtol=1e-6)
    assert row.water_valid_fraction == row.et_valid_fraction == 1
