"""Small live PC/GEE input-transfer benchmark, isolated from production data.

Three acquisitions per sensor, all required bands and QA, two window sizes.
Every measurement runs in a fresh process. PC uses four concurrent COG readers;
GEE uses one computePixels request. Setup/search timings are recorded separately.
S1 compares delivery workflows, not equivalent radiometric products.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

S2_BANDS = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06',
            'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
LS_BANDS = ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
COLLECTIONS = {'landsat': 'landsat-c2-l2', 's2': 'sentinel-2-l2a',
               's1': 'sentinel-1-rtc'}
MONTHS = ['2025-01', '2025-04', '2025-07']


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def fetch_json(url, **kwargs):
    r = requests.get(url, timeout=45, **kwargs)
    r.raise_for_status()
    return r.json()


def initialize_ee(project):
    import ee
    ee.Initialize(project=project)
    ee.data.setDeadline(60000)
    return ee


def prepare(root, project):
    """Fetch small catalog pages and resolve same-acquisition EE counterparts."""
    jobs, searches = [], []
    for month in MONTHS:
        for sensor, collection in COLLECTIONS.items():
            start = month + '-01T00:00:00Z'
            end = month + '-28T23:59:59Z'
            params = dict(collections=collection, bbox='116.14,29.17,116.16,29.19',
                          datetime=start + '/' + end, limit=100)
            if sensor == 'landsat':
                params['query'] = json.dumps({
                    'platform': {'in': ['landsat-8', 'landsat-9']},
                    'landsat:collection_category': {'eq': 'T1'}})
            t = time.perf_counter()
            result = fetch_json('https://planetarycomputer.microsoft.com/api/stac/v1/search',
                                params=params)
            items = result['features']
            if sensor == 's2':
                items = [x for x in items if x['id'].startswith(('S2A_', 'S2B_'))]
            if sensor == 's1':
                items = [x for x in items if x['id'].startswith('S1A_IW_GRDH_1SDV_')
                         and {'vv', 'vh'} <= x['assets'].keys()]
            if not items:
                raise RuntimeError(f'No {sensor} candidates for {month}')
            from shapely.geometry import Point, shape
            items = [x for x in items if shape(x['geometry']).covers(Point(116.15, 29.18))]
            item = min(items, key=lambda x: (x['properties'].get('eo:cloud_cover', 0),
                                             x['id']))
            searches.append(dict(sensor=sensor, month=month, seconds=time.perf_counter() - t,
                                 returned_candidates=len(items)))
            jobs.append(dict(sensor=sensor, month=month, item=item))
            print('selected', sensor, month, item['id'], flush=True)
    t = time.perf_counter()
    ee = initialize_ee(project)
    resolutions = {}
    for i, job in enumerate(jobs):
        sensor, item = job['sensor'], job['item']
        if sensor == 'landsat':
            mission, _, wrs, day, *_ = item['id'].split('_')
            c = ee.ImageCollection(f'LANDSAT/{mission}/C02/T1_L2').filter(
                ee.Filter.eq('system:index', f'{mission}_{wrs}_{day}'))
        elif sensor == 's1':
            prefix = item['id'].removesuffix('_rtc')
            day = datetime.strptime(item['id'].split('_')[4][:8], '%Y%m%d')
            c = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterDate(day.strftime('%Y-%m-%d'),
                             (day + timedelta(days=1)).strftime('%Y-%m-%d'))
                 .filter(ee.Filter.stringStartsWith('system:index', prefix)))
        else:
            day = datetime.strptime(item['id'].split('_')[2][:8], '%Y%m%d')
            c = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterDate(day.strftime('%Y-%m-%d'),
                             (day + timedelta(days=1)).strftime('%Y-%m-%d'))
                 .filter(ee.Filter.eq('MGRS_TILE', item['properties']['s2:mgrs_tile']))
                 .filter(ee.Filter.eq('SPACECRAFT_NAME',
                                     'Sentinel-2' + item['id'][2])))
            exact = c.filter(ee.Filter.eq('PRODUCT_ID', item['properties']['s2:product_uri']))
            c = ee.ImageCollection(ee.Algorithms.If(exact.size().gt(0), exact, c))
        resolutions[str(i)] = c.limit(3).aggregate_array('system:id')
    resolved = ee.Dictionary(resolutions).getInfo()
    for i, job in enumerate(jobs):
        ids = resolved[str(i)]
        if not ids:
            raise RuntimeError(f"No EE counterpart for {job['item']['id']}")
        job['gee_asset'] = sorted(ids)[-1]
        job['gee_candidates'] = ids
    from rasterio.warp import transform
    x, y = transform('EPSG:4326', 'EPSG:32650', [116.15], [29.18])
    manifest = dict(
        created_utc=datetime.now(UTC).isoformat(), project=project,
        host=os.uname().nodename, center_lonlat=[116.15, 29.18],
        center_xy=[round(x[0] / 10) * 10, round(y[0] / 10) * 10],
        crs='EPSG:32650', pixel_size_m=10, sizes=[120, 480],
        pc_band_workers=4, pc_searches=searches,
        gee_resolution_and_initialization_s=time.perf_counter() - t, jobs=jobs,
        caveats=[
            'Single acquisition per window; not a complete four-slot training example.',
            'S1: PC RTC gamma0 dB versus GEE sigma0 dB; transport comparison only.',
            'Fresh process per measurement; upstream caches and network routing uncontrolled.',
            'No target labels, sampling, full time-series selection or model inference timed.',
            'PC four concurrent band readers; GEE one request with all bands.',
            'S2 same acquisition may have different processing versions.',
            'Three clear-ish acquisitions over one Poyang location, not global throughput.'
        ])
    save(root / 'manifest.json', manifest)
    return manifest


def pc_setup(item, sensor):
    names = LS_BANDS + ['qa_pixel', 'qa_radsat'] if sensor == 'landsat' else (
        S2_BANDS + ['SCL'] if sensor == 's2' else ['vv', 'vh'])
    tokens, urls = {}, {}
    for name in [*names, *(['product-metadata'] if sensor == 's2' else [])]:
        href = item['assets'][name]['href']
        parsed = urlsplit(href)
        account, container = parsed.netloc.split('.')[0], parsed.path.split('/')[1]
        if (account, container) not in tokens:
            data = fetch_json(
                f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}')
            tokens[account, container] = data['token']
        urls[name] = href.split('?')[0] + '?' + tokens[account, container]
    offsets, quantification = {}, 10000.0
    if sensor == 's2':
        response = requests.get(urls['product-metadata'], timeout=45)
        response.raise_for_status()
        tags = {}
        for e in ET.fromstring(response.content).iter():
            tags.setdefault(e.tag.rsplit('}', 1)[-1], []).append(e)
        quantification = float(tags['BOA_QUANTIFICATION_VALUE'][0].text)
        ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
               'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        offsets = {ids[int(e.attrib['band_id'])]: float(e.text)
                   for e in tags.get('BOA_ADD_OFFSET', [])}
        if float(tags['PROCESSING_BASELINE'][0].text) >= 4:
            if any(b not in offsets for b in S2_BANDS):
                raise ValueError('Missing S2 processing offsets')
    return names, urls, offsets, quantification


def pc_read(sensor, setup, grid, size):
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    from hydrostate.data.input_products import harmonize_s2, landsat_l2_pixels
    names, urls, offsets, quantification = setup

    def band(name):
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          GDAL_HTTP_CONNECTTIMEOUT='15', GDAL_HTTP_TIMEOUT='45',
                          GDAL_HTTP_MAX_RETRY='1', GDAL_CACHEMAX=32 * 1024 * 1024):
            with rasterio.open(urls[name]) as src:
                with WarpedVRT(src, crs='EPSG:32650', transform=Affine(*grid),
                               width=size, height=size, dtype='float32',
                               nodata=float('nan'), resampling=Resampling.nearest,
                               warp_mem_limit=16) as vrt:
                    return vrt.read(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        arrays = dict(zip(names, pool.map(band, names)))
    band_names = LS_BANDS if sensor == 'landsat' else (
        S2_BANDS if sensor == 's2' else ['vv', 'vh'])
    data = np.stack([arrays[x] for x in band_names])
    valid = np.isfinite(data).all(axis=0)
    if sensor == 'landsat':
        qa, sat = arrays['qa_pixel'], arrays['qa_radsat']
        valid &= np.isfinite(qa) & np.isfinite(sat)
        data, valid = landsat_l2_pixels(data, np.nan_to_num(qa),
                                       np.nan_to_num(sat), valid)
    elif sensor == 's2':
        valid &= (data != 0).all(axis=0)
        valid &= np.isfinite(arrays['SCL']) & ~np.isin(arrays['SCL'], [0, 1, 3, 8, 9, 10])
        data = np.stack([harmonize_s2(a, offset=offsets.get(b, 0),
                                     quantification=quantification)
                         for b, a in zip(S2_BANDS, data)])
    else:
        valid &= (data > 0).all(axis=0)
        data = 10 * np.log10(np.maximum(data, np.finfo('float32').tiny))
    return np.where(valid[None], data, 0).astype('float32'), valid


def gee_expression(ee, job):
    image = ee.Image(job['gee_asset'])
    sensor = job['sensor']
    if sensor == 'landsat':
        names = [f'SR_B{i}' for i in range(1, 8)]
        raw = image.select(names)
        valid = (raw.mask().reduce(ee.Reducer.min()).gt(0)
                 .And(raw.neq(0).reduce(ee.Reducer.min()))
                 .And(image.select('QA_PIXEL').bitwiseAnd(63).eq(0))
                 .And(image.select('QA_RADSAT').eq(0)))
        data = raw.multiply(0.0000275).add(-0.2)
    elif sensor == 's2':
        names = ['B' + str(int(x[1:])) if x != 'B8A' else x for x in S2_BANDS]
        data = image.select(names)
        valid = data.mask().reduce(ee.Reducer.min()).gt(0)
        for value in [0, 1, 3, 8, 9, 10]:
            valid = valid.And(image.select('SCL').neq(value))
    else:
        names = ['VV', 'VH']
        data = image.select(names)
        valid = data.mask().reduce(ee.Reducer.min()).gt(0)
    expression = ee.Image.cat([data.updateMask(valid).unmask(0, sameFootprint=False),
                               valid.unmask(0, sameFootprint=False).rename('valid')]).toFloat()
    return expression, names + ['valid']


def worker(root, index, provider, size):
    logging.getLogger('rasterio').setLevel(logging.CRITICAL)
    manifest = json.loads((root / 'manifest.json').read_text())
    job = manifest['jobs'][index]
    identity = f"{index:02d}_{job['sensor']}_{job['month']}_{provider}_{size}"
    result = dict(identity=identity, provider=provider, sensor=job['sensor'],
                  month=job['month'], size=size, pc_item=job['item']['id'],
                  gee_asset=job['gee_asset'])
    x, y = job.get('center_xy', manifest['center_xy'])
    result['center_xy'] = [x, y]
    grid = [10, 0, x - size * 5, 0, -10, y + size * 5]
    t = time.perf_counter()
    try:
        if provider == 'pc':
            setup = pc_setup(job['item'], job['sensor'])
        else:
            ee = initialize_ee(manifest['project'])
            expression, names = gee_expression(ee, job)
        result['setup_s'] = time.perf_counter() - t
        started = time.perf_counter()
        if provider == 'pc':
            data, valid = pc_read(job['sensor'], setup, grid, size)
        else:
            payload = ee.data.computePixels(dict(
                expression=expression, fileFormat='NUMPY_NDARRAY', bandIds=names,
                grid=dict(dimensions=dict(width=size, height=size), crsCode='EPSG:32650',
                          affineTransform=dict(zip(
                              ['scaleX', 'shearX', 'translateX', 'shearY',
                               'scaleY', 'translateY'], grid)))))
            data = np.stack([payload[b] for b in names[:-1]]).astype('float32')
            valid = payload['valid'] > 0
        result['read_process_s'] = time.perf_counter() - started
        if data.shape[1:] != (size, size) or not np.isfinite(data).all():
            raise ValueError('Malformed data')
        started = time.perf_counter()
        filename = root / 'arrays' / (identity + '.npz')
        with filename.open('wb') as f:
            np.savez(f, values=data, valid=valid, transform=np.asarray(grid))
            f.flush()
            os.fsync(f.fileno())
        with np.load(filename) as check:
            assert np.array_equal(check['values'], data)
            assert np.array_equal(check['valid'], valid)
        result.update(write_verify_s=time.perf_counter() - started,
                      output_bytes=filename.stat().st_size,
                      valid_fraction=float(valid.mean()), band_count=data.shape[0],
                      sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                      total_s=time.perf_counter() - t, status='ok')
    except Exception as e:
        # Never persist or print signed URLs from HTTP/GDAL exception messages.
        result.update(status='failed', error_type=type(e).__name__,
                      elapsed_s=time.perf_counter() - t)
    save(root / 'results' / (identity + '.json'), result)
    print(json.dumps(result), flush=True)


def summarize(root):
    results = [json.loads(p.read_text()) for p in sorted((root / 'results').glob('*.json'))]
    # Only report timing pairs where both providers delivered >=50% valid pixels.
    eligible_pairs = set()
    for row in results:
        prefix = row['identity'].split('_')[0]
        pair = [r for r in results if r['identity'].split('_')[0] == prefix
                and r['size'] == row['size']]
        if ({r['provider'] for r in pair} == {'pc', 'gee'} and
                all(r['status'] == 'ok' and r.get('valid_fraction', 0) >= .5 for r in pair)):
            eligible_pairs.add((prefix, row['size']))
    groups = []
    for sensor in COLLECTIONS:
        for size in [120, 480]:
            for provider in ['pc', 'gee']:
                rows = [r for r in results if (r['sensor'], r['size'], r['provider']) ==
                        (sensor, size, provider)]
                ok = [r for r in rows if (r['identity'].split('_')[0], size) in eligible_pairs]
                g = dict(sensor=sensor, size=size, provider=provider,
                         successes=sum(r['status'] == 'ok' for r in rows),
                         eligible_measurements=len(ok), attempts=len(rows))
                for k in ['setup_s', 'read_process_s', 'write_verify_s', 'total_s',
                          'valid_fraction']:
                    if ok:
                        g[k + '_median'] = float(np.median([r[k] for r in ok]))
                        g[k + '_range'] = [min(r[k] for r in ok), max(r[k] for r in ok)]
                groups.append(g)
    save(root / 'summary.json', dict(
        eligibility='Both matched providers succeeded and each had >=50% valid pixels.',
        groups=groups, results=results))
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'summarize'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--project', default='cropland-477906')
    parser.add_argument('--index', type=int)
    parser.add_argument('--provider', choices=['pc', 'gee'])
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['arrays', 'results', 'logs']:
        (root / folder).mkdir(exist_ok=True)
    if args.command == 'prepare':
        if (root / 'manifest.json').exists():
            raise FileExistsError('Use a new output directory')
        prepare(root, args.project)
    elif args.command == 'worker':
        worker(root, args.index, args.provider, args.size)
    elif args.command == 'summarize':
        summarize(root)
    else:
        manifest = json.loads((root / 'manifest.json').read_text())
        for index, job in enumerate(manifest['jobs']):
            for size in manifest['sizes']:
                # Alternate provider order to reduce systematic order bias.
                order = ['pc', 'gee'] if (index + size // 120) % 2 else ['gee', 'pc']
                for provider in order:
                    identity = f"{index:02d}_{job['sensor']}_{job['month']}_{provider}_{size}"
                    dest = root / 'results' / (identity + '.json')
                    if dest.exists():
                        continue
                    print('running', identity, flush=True)
                    command = [sys.executable, __file__, 'worker', '--output', str(root),
                               '--index', str(index), '--provider', provider, '--size', str(size)]
                    with (root / 'logs' / (identity + '.log')).open('w') as log:
                        try:
                            subprocess.run(command, stdout=log, stderr=log, timeout=150, check=True)
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                            save(dest, dict(identity=identity, sensor=job['sensor'],
                                            month=job['month'], size=size, provider=provider,
                                            status='failed', error_type=type(e).__name__))
                    if dest.exists():
                        result = json.loads(dest.read_text())
                        print(json.dumps({k: result.get(k) for k in
                                          ['identity', 'status', 'read_process_s',
                                           'total_s', 'valid_fraction']}), flush=True)
        summarize(root)


if __name__ == '__main__':
    main()
