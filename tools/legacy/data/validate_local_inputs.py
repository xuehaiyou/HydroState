"""Read real local products and optionally compare identical acquisitions from GEE."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from pyproj import Transformer

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.contracts import input_transform
from hydrostate.data.gee.products import scene_expression
from hydrostate.data.input_products import LANDSAT_SR_BANDS, landsat_l2_pixels
from hydrostate.data.local.catalog import inspect_scene
from hydrostate.data.local.reader import RasterReader


def gee_pixels(ee, scene, grid):
    name = Path(scene['asset_id'].removeprefix('local:')).name
    if scene['key'] == 'landsat':
        sensor, _, wrs, day = name.split('_')[:4]
        asset = f'LANDSAT/{sensor}/C02/T1_L2/{sensor}_{wrs}_{day}'
        image = ee.Image(asset).select([*LANDSAT_SR_BANDS, 'QA_PIXEL', 'QA_RADSAT'])
        valid = image.mask().reduce(ee.Reducer.min()).gt(0).rename('valid')
        expression = ee.Image.cat([image, valid]).unmask(0, sameFootprint=False).toFloat()
        names = [*LANDSAT_SR_BANDS, 'QA_PIXEL', 'QA_RADSAT', 'valid']
    else:
        from hydrostate.data.gee.products import BANDS

        if scene['key'] == 's1':
            asset = 'COPERNICUS/S1_GRD/' + name
        else:
            tile = re.search(r'_T(\d{2}[A-Z]{3})_', name).group(1)
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filter(ee.Filter.eq('MGRS_TILE', tile))
                          .filterDate(ee.Date(scene['time'] * 1000 - 60000),
                                      ee.Date(scene['time'] * 1000 + 60000)))
            candidates = collection.aggregate_array('system:id').getInfo()
            if len(candidates) != 1:
                raise ValueError(f'Expected one matching GEE S2 acquisition: {candidates}')
            asset = candidates[0]
        expression = scene_expression(ee, dict(key=scene['key'], asset_id=asset))
        names = [*BANDS[scene['key']], 'valid']
    raw = compute_pixels(ee, expression, names, grid['crs'],
                         list(input_transform(grid))[:6], 120, 120)
    if scene['key'] == 'landsat':
        values, valid = landsat_l2_pixels(np.stack([raw[b] for b in LANDSAT_SR_BANDS]),
                                          raw['QA_PIXEL'], raw['QA_RADSAT'], raw['valid'] > 0)
    else:
        values, valid = np.stack([raw[b] for b in names[:-1]]), raw['valid'] > 0
    return asset, values, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['s1', 's2', 'landsat']:
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--crs', default='EPSG:32650')
    parser.add_argument('--compare-gee', action='store_true')
    parser.add_argument('--project', default='cropland-477906')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any(getattr(args, key) for key in ['s1', 's2', 'landsat']):
        parser.error('At least one local product is required')
    x, y = Transformer.from_crs(4326, args.crs, always_xy=True).transform(args.lon, args.lat)
    grid = dict(crs=args.crs, left=round(x / 10) * 10 - 480,
                top=round(y / 10) * 10 + 480, height=32, width=32, halo=4, core=32)
    reader = RasterReader(grid)
    ee = None
    if args.compare_gee:
        from hydrostate.data.gee.client import initialize

        ee = initialize(args.project)
    report = dict(grid=grid, comparison='descriptive;not_an_automatic_equivalence_certificate',
                  sensors={})
    for key in ['s1', 's2', 'landsat']:
        if not getattr(args, key):
            continue
        scene = inspect_scene(key, getattr(args, key), args.crs)
        values, mask = reader.read(scene, 0, 0, 120, 120)
        record = dict(source=scene, valid_fraction=float(mask.mean()),
                      finite=bool(np.isfinite(values).all()), shape=list(values.shape))
        if ee is not None:
            asset, other, valid = gee_pixels(ee, scene, grid)
            shared = mask & valid
            record.update(gee_asset=asset, gee_valid_fraction=float(valid.mean()),
                          shared_pixels=int(shared.sum()),
                          mask_agreement=float((valid == mask).mean()))
            if shared.any():
                error = values[:, shared].astype('float64') - other[:, shared]
                record.update(bias_per_band=error.mean(axis=1).tolist(),
                              rmse_per_band=np.sqrt((error ** 2).mean(axis=1)).tolist(),
                              max_abs_per_band=np.abs(error).max(axis=1).tolist())
        report['sensors'][key] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {name: value for name, value in record.items() if name != 'source'}
                      for k, record in report['sensors'].items()}, indent=2))


if __name__ == '__main__':
    main()
