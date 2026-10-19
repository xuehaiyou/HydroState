"""Cross-source units, masks, immutable packs and offline acquisition integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box, mapping

from hydrostate.data.gee.contracts import input_transform
from hydrostate.data.gee.products import BANDS, normalize_pixels
from hydrostate.data.gee.workflows.landsat_l2 import l2_asset
from hydrostate.data.gee.workflows.region import assemble_inputs
from hydrostate.data.input_products import L2_PROFILE, harmonize_s2, landsat_l2_pixels
from hydrostate.data.local.catalog import inspect_scene, s2_metadata
from hydrostate.data.local.reader import RasterReader
from hydrostate.data.local.region import prepare_region_local
from hydrostate.storage.scene_cache import SceneCache
from tools.data.extend_gee_labels import Checkpoint
from tools.data.extend_gee_landsat_l2 import publish


def test_sr_quality_preserves_water_and_negative_reflectance():
    dn = np.full((7, 2, 4), 5000, 'uint16')
    qa = np.array([[128, 0, 8, 32], [1, 2, 4, 16]], 'uint16')
    values, valid = landsat_l2_pixels(dn, qa, np.zeros_like(qa))
    assert valid.tolist() == [[True, True, False, False], [False] * 4]
    np.testing.assert_allclose(values[:, 0, :2], -0.0625)
    assert (values[:, ~valid] == 0).all()
    dn[3, 0, 0] = 0
    assert not landsat_l2_pixels(dn, qa, np.zeros_like(qa))[1][0, 0]
    saturated = np.zeros_like(qa)
    saturated[0, 1] = 1
    assert not landsat_l2_pixels(dn, qa, saturated)[1].any()


def test_assets_must_match_original_l1_collection():
    assert l2_asset('LANDSAT/LC08/C02/T1/LC08_121040_20170712') == (
        'LANDSAT/LC08/C02/T1_L2/LC08_121040_20170712')
    with pytest.raises(ValueError):
        l2_asset('LANDSAT/LC08/C02/T2/LC08_121040_20170712')


def raster(path, value, tr, *, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(value)
    if data.ndim == 0:
        data = np.full((120, 120), value, dtype='float32')
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                       count=1, dtype=data.dtype, transform=tr, crs='EPSG:32650',
                       nodata=nodata) as dst:
        dst.write(data, 1)


def sources(root, tr, day='20200715'):
    landsat = root / 'landsat' / f'LC08_L2SP_121040_{day}_20200903_02_T1'
    for i in range(1, 8):
        raster(landsat / f'test_SR_B{i}.TIF', 10000, tr, nodata=0)
    qa = np.full((120, 120), 128, 'uint16')  # Keep water.
    qa[0, 0] = 8
    raster(landsat / 'test_QA_PIXEL.TIF', qa, tr)
    raster(landsat / 'test_QA_RADSAT.TIF', 0, tr)
    mtl = dict(IMAGE_ATTRIBUTES=dict(SPACECRAFT_ID='LANDSAT_8',
                                    DATE_ACQUIRED=f'{day[:4]}-{day[4:6]}-{day[6:]}',
                                    SCENE_CENTER_TIME='02:44:09.0050290Z'),
               PRODUCT_CONTENTS=dict(PROCESSING_LEVEL='L2SP', COLLECTION_CATEGORY='T1'),
               LEVEL2_SURFACE_REFLECTANCE_PARAMETERS={
                   **{f'REFLECTANCE_MULT_BAND_{i}': 0.0000275 for i in range(1, 8)},
                   **{f'REFLECTANCE_ADD_BAND_{i}': -0.2 for i in range(1, 8)}})
    (landsat / 'test_MTL.json').write_text(json.dumps(dict(LANDSAT_METADATA_FILE=mtl)))
    s2 = root / 's2' / f'S2A_MSIL2A_{day}T025941_N0500_R032_T50RMT_20230101T000000.SAFE'
    s2.mkdir(parents=True)
    offsets = ''.join(f'<BOA_ADD_OFFSET band_id="{i}">-1000</BOA_ADD_OFFSET>' for i in range(13))
    (s2 / 'MTD_MSIL2A.xml').write_text('<root><PROCESSING_BASELINE>05.00</PROCESSING_BASELINE>'
        '<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>' + offsets + '</root>')
    granule = s2 / 'GRANULE' / 'tile'
    granule.mkdir(parents=True)
    (granule / 'MTD_TL.xml').write_text(
        f'<root><SENSING_TIME>{day[:4]}-{day[4:6]}-{day[6:]}T02:59:41Z</SENSING_TIME></root>')
    for b in BANDS['s2']:
        res = 10 if b in ['B2', 'B3', 'B4', 'B8'] else 60 if b in ['B1', 'B9'] else 20
        name = b if b == 'B8A' else 'B' + b[1:].zfill(2)
        raster(granule / f'IMG_DATA/R{res}m/test_{name}_{res}m.jp2', 3000, tr, nodata=0)
    raster(granule / 'IMG_DATA/R20m/test_SCL_20m.jp2', 6, tr)
    s1 = root / 's1' / f'S1A_IW_GRDH_1SDV_{day}T101851_{day}T101916_036337_044382_61F8'
    for b, value in [('VV', -12), ('VH', -18)]:
        raster(s1 / f'Sigma0_{b}_db.tif', value, tr, nodata=-999)
    return dict(s1=s1, s2=s2, landsat=landsat)


@pytest.fixture
def local_setup(tmp_path):
    grid = dict(crs='EPSG:32650', left=400000, top=3200000,
                height=32, width=32, halo=4, core=32)
    plan = dict(grid=grid, training=[], geometry=mapping(box(400000, 3199040, 400960, 3200000)),
                config=dict(input_profile=L2_PROFILE, cache_chunk=120, min_valid_fraction=0.5,
                            min_active_slots=2, latest_days=4),
                periods=[dict(time='2020-07-15', start='2020-07-11', end='2020-07-19')])
    paths = sources(tmp_path / 'sources', input_transform(grid))
    sources(tmp_path / 'sources', input_transform(grid), '20200629')
    settings = dict(s1_units='sigma0_db', roots={k: [str(p.parent)] for k, p in paths.items()})
    return plan, settings, paths


def test_local_read_matches_physical_and_normalized_input_contract(tmp_path, local_setup):
    plan, _, paths = local_setup
    reader = RasterReader(plan['grid'])
    cache = SceneCache(tmp_path, plan)
    for key, path in paths.items():
        scene = inspect_scene(key, path, plan['grid']['crs'])
        values, valid = reader.read(scene, 0, 0, 120, 120)
        if key == 'landsat':
            assert not valid[0, 0] and valid[0, 1]
            np.testing.assert_allclose(values[:, 0, 1], 0.075, atol=1e-7)
        elif key == 's2':
            assert valid.all()
            np.testing.assert_array_equal(values, 2000)
        cache.write_chunk(scene, 0, 0, values, valid)
        stored, mask = cache.read(scene, dict(row=0, col=0))
        expected = (values if key == 'landsat'
                    else normalize_pixels(key, values, valid, cache.norm))
        np.testing.assert_array_equal(stored, expected)
        np.testing.assert_array_equal(mask, valid)


def test_full_local_backend_publishes_only_complete_inputs_and_resumes(tmp_path, local_setup,
                                                                    monkeypatch):
    plan, settings, _ = local_setup
    root = tmp_path / 'region'
    prepare_region_local(root, plan, plan['periods'], settings, workers=2, min_free_gib=5)
    table = pd.read_parquet(root / 'manifests/2020-07-15.parquet')
    assert len(table) == 1 and table.accepted.all()
    cache = SceneCache(root, plan)
    arrays = assemble_inputs(cache, table.iloc[0])
    assert 'landsat' not in arrays
    assert arrays['landsat_l2'].shape == (4, 7, 120, 120)
    assert arrays['landsat_l2_valid'].shape == arrays['landsat_l2'].shape
    assert arrays['landsat_l2_time'][3] == arrays['observation_time'][3, 2]
    assert arrays['sensor_valid'][2:].all() and not arrays['sensor_valid'][:2].any()
    monkeypatch.setattr(RasterReader, 'read', lambda *a: pytest.fail('Resume decoded source again'))
    prepare_region_local(root, plan, plan['periods'], settings, workers=2, min_free_gib=5)
    changed = dict(plan, config=dict(plan['config'], latest_days=10))
    with pytest.raises(ValueError, match='configuration changed'):
        prepare_region_local(root, changed, plan['periods'], settings, min_free_gib=5)


def test_s2_baseline_not_acquisition_year_and_requires_offsets(local_setup):
    _, _, paths = local_setup
    params, _, _ = s2_metadata(paths['s2'])
    assert params['offsets']['B2'] == -1000
    np.testing.assert_array_equal(harmonize_s2(np.array([500, 3000]), offset=-1000,
                                             quantification=10000), [0, 2000])
    path = paths['s2'] / 'MTD_MSIL2A.xml'
    path.write_text(path.read_text().replace(
        '<BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET>', ''))
    with pytest.raises(ValueError, match='Missing BOA offsets'):
        s2_metadata(paths['s2'])


def test_source_mutation_rejected_before_read(local_setup):
    plan, _, paths = local_setup
    scene = inspect_scene('s1', paths['s1'], plan['grid']['crs'])
    Path(scene['files']['VV']).touch()
    with pytest.raises(ValueError, match='Indexed source changed'):
        RasterReader(plan['grid']).read(scene, 0, 0, 120, 120)


def test_pack_publish_retains_identity_missing_slots_and_times(tmp_path):
    from hydrostate.storage.packs import VariablePack, file_hash

    table = pd.DataFrame([dict(sample_id='a', split='train', target_time='2020-07-15')])
    manifest = tmp_path / 'samples.parquet'
    table.to_parquet(manifest, index=False)
    values = dict(landsat_l2=np.zeros((4, 7, 120, 120), 'float32'),
                  landsat_l2_valid=np.zeros((4, 7, 120, 120), bool),
                  landsat_l2_time=np.array([0, 0, 0, 1594771200], 'int64'))
    values['landsat_l2'][3] = 0.075
    values['landsat_l2_valid'][3] = True
    contract = dict(manifest_sha256=file_hash(manifest))
    cache = Checkpoint(tmp_path / 'checkpoint.sqlite', contract)
    cache.save('a', values, dict(missing_l2_slots=[2]))
    args = SimpleNamespace(output=tmp_path / 'pack', manifest=manifest)
    report = publish(table.to_dict('records'), cache, args, contract)
    cache.close()
    assert report['valid_slots'] == report['samples_with_l2'] == 1
    pack = VariablePack(args.output, table, kind='input')
    import zarr

    stored = pack.read('a', SimpleNamespace(get=lambda p: zarr.open_group(p, mode='r')))
    for name in values:
        np.testing.assert_array_equal(stored[name], values[name])


def test_catalog_skips_t2_before_requiring_bands(tmp_path, local_setup):
    from hydrostate.data.local.catalog import ExcludedScene, catalog

    plan, settings, paths = local_setup
    t2 = paths['landsat'].with_name(paths['landsat'].name.replace('_T1', '_T2'))
    t2.mkdir()
    metadata = json.loads((paths['landsat'] / 'test_MTL.json').read_text())
    metadata['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['COLLECTION_CATEGORY'] = 'T2'
    # Only the MTL exists: exclusion must happen before opening any QA or raster.
    (t2 / 'test_MTL.json').write_text(json.dumps(metadata))
    with pytest.raises(ExcludedScene, match='landsat_tier_2'):
        inspect_scene('landsat', t2, plan['grid']['crs'])
    root = tmp_path / 'region'
    root.mkdir()
    result = catalog(root, plan, settings)
    assert len(result['scenes']) == 6
    assert result['excluded'] == [dict(key='landsat', path=str(t2), reason='landsat_tier_2')]
    progress = json.loads((root / 'local_catalog_progress.json').read_text())
    assert progress['status'] == 'complete'
    assert progress['sensors']['landsat']['excluded'] == 1
    assert progress['sensors']['landsat']['accepted'] == 2


def test_catalog_resumes_completed_scenes_and_discovery_after_failure(tmp_path, local_setup,
                                                                   monkeypatch):
    import importlib
    import sqlite3

    module = importlib.import_module('hydrostate.data.local.catalog')
    plan, settings, _ = local_setup
    root = tmp_path / 'region'
    root.mkdir()
    inspected = []

    def interrupted(key, path, crs):
        inspected.append((key, str(path)))
        if len(inspected) == 6:
            raise OSError('Simulated last-scene read failure')
        return inspect_scene(key, path, crs)

    monkeypatch.setattr(module, 'inspect_scene', interrupted)
    with pytest.raises(OSError, match='last-scene'):
        module.catalog(root, plan, settings)
    assert not (root / 'local_catalog.json').exists()
    with sqlite3.connect(root / 'local_catalog.sqlite') as db:
        assert db.execute('SELECT count(*) FROM scenes').fetchone()[0] == 5
        assert db.execute('SELECT count(*) FROM discoveries').fetchone()[0] == 3
    failed = json.loads((root / 'local_catalog_progress.json').read_text())
    assert failed['status'] == 'failed' and failed['path'] == inspected[-1][1]

    def resume(key, path, crs):
        assert (key, str(path)) == inspected[-1]
        return inspect_scene(key, path, crs)

    monkeypatch.setattr(module, 'inspect_scene', resume)
    monkeypatch.setattr(module, 'discover', lambda *a: pytest.fail('Repeated directory discovery'))
    result = module.catalog(root, plan, settings)
    assert len(result['scenes']) == 6
    progress = json.loads((root / 'local_catalog_progress.json').read_text())
    assert progress['status'] == 'complete'
    assert sum(v['resumed'] for v in progress['sensors'].values()) == 5
    assert sum(v['inspected'] for v in progress['sensors'].values()) == 1


def test_catalog_keeps_unknown_tiers_as_errors_and_rejects_changed_checkpoint(tmp_path,
                                                                          local_setup):
    from hydrostate.data.local.catalog import catalog

    plan, settings, paths = local_setup
    path = paths['landsat'] / 'test_MTL.json'
    metadata = json.loads(path.read_text())
    metadata['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS'].pop('COLLECTION_CATEGORY')
    path.write_text(json.dumps(metadata))
    root = tmp_path / 'region'
    root.mkdir()
    with pytest.raises(ValueError, match='Tier 1'):
        catalog(root, plan, settings)
    assert not (root / 'local_catalog.json').exists()
    changed = dict(plan, config=dict(plan['config'], latest_days=10))
    with pytest.raises(ValueError, match='checkpoint configuration changed'):
        catalog(root, changed, settings)
    # Previously indexed scenes must still refer to the frozen source files.
    (paths['s1'] / 'Sigma0_VV_db.tif').touch()
    with pytest.raises(ValueError, match='Indexed source changed'):
        catalog(root, plan, settings)
