"""An explicit, frozen regional index of source files and processing metadata."""

import glob
import json
import logging
import re
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from hydrostate.data.gee.contracts import atomic_json, fingerprint, input_shape, input_transform
from hydrostate.data.gee.products import BANDS
from hydrostate.data.input_products import LANDSAT_SR_BANDS, LANDSAT_SR_OFFSET, LANDSAT_SR_SCALE
from hydrostate.data.local.index import CatalogCheckpoint
from hydrostate.data.regional.temporal import regional_slots

VERSION = 'local_l2_catalog_v1'
LOG = logging.getLogger(__name__)


class ExcludedScene(ValueError):
    """An explicitly excluded product, not a corrupt or incomplete source."""

    def __init__(self, reason, metadata):
        super().__init__(f'{reason}: {metadata.parent}')
        self.reason = reason
        stat = metadata.stat()
        self.snapshot = {str(metadata): [stat.st_size, stat.st_mtime_ns]}


def timestamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00'))
               .replace(tzinfo=UTC).timestamp())


def one(paths, description):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f'{description}: expected exactly one file, found {len(paths)}')
    return paths[0]


def xml_values(path):
    root = ET.parse(path).getroot()
    values = {}
    for element in root.iter():
        values.setdefault(element.tag.rsplit('}', 1)[-1], []).append(element)
    return values


def s2_metadata(safe):
    path = safe / 'MTD_MSIL2A.xml'
    tags = xml_values(path)
    baseline = float(tags['PROCESSING_BASELINE'][0].text)
    quantification = float(tags['BOA_QUANTIFICATION_VALUE'][0].text)
    identifiers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                   'B8A', 'B9', 'B10', 'B11', 'B12']
    offsets = {identifiers[int(e.attrib['band_id'])]: float(e.text)
               for e in tags.get('BOA_ADD_OFFSET', [])}
    if baseline >= 4 and any(b not in offsets for b in BANDS['s2']):
        raise ValueError(f'Missing BOA offsets for baseline {baseline}: {safe}')
    granule = one((safe / 'GRANULE').glob('*/MTD_TL.xml'), 'S2 granule metadata')
    sensing = xml_values(granule)['SENSING_TIME'][0].text
    return dict(baseline=baseline, quantification=quantification,
                offsets={b: offsets.get(b, 0) for b in BANDS['s2']}), sensing, [path, granule]


def inspect_scene(key, path, crs):
    """Strictly reject incomplete/ambiguous products; never guess missing QA."""
    path = Path(path)
    parameters, metadata = {}, []
    if key == 'landsat':
        mtl = one(path.glob('*_MTL.json'), 'Landsat MTL')
        data = json.loads(mtl.read_text())['LANDSAT_METADATA_FILE']
        attributes = data['IMAGE_ATTRIBUTES']
        content = data['PRODUCT_CONTENTS']
        if content['PROCESSING_LEVEL'] not in {'L2SP', 'L2SR'}:
            raise ValueError(f'Not a Landsat L2 scene: {path}')
        if attributes.get('SPACECRAFT_ID') not in {'LANDSAT_8', 'LANDSAT_9'}:
            raise ValueError(f'Only Landsat 8/9 are supported: {path}')
        category = attributes.get('COLLECTION_CATEGORY', content.get('COLLECTION_CATEGORY'))
        if category == 'T2':
            raise ExcludedScene('landsat_tier_2', mtl)
        if category != 'T1':
            raise ValueError(f'Only Landsat Collection 2 Tier 1 is supported: {path}')
        scaling = data['LEVEL2_SURFACE_REFLECTANCE_PARAMETERS']
        for i in range(1, 8):
            if (abs(float(scaling[f'REFLECTANCE_MULT_BAND_{i}']) - LANDSAT_SR_SCALE) > 1e-12
                    or abs(float(scaling[f'REFLECTANCE_ADD_BAND_{i}'])
                           - LANDSAT_SR_OFFSET) > 1e-9):
                raise ValueError(f'Unexpected Landsat SR scaling: {path}')
        files = {b: one(path.glob(f'*_{b}.TIF'), b)
                 for b in [*LANDSAT_SR_BANDS, 'QA_PIXEL', 'QA_RADSAT']}
        acquired = attributes['DATE_ACQUIRED'] + 'T' + attributes['SCENE_CENTER_TIME']
        metadata = [mtl]
        orbit = None
    elif key == 's2':
        parameters, acquired, metadata = s2_metadata(path)
        resolutions = {b: 20 for b in BANDS['s2']}
        resolutions.update(B2=10, B3=10, B4=10, B8=10, B1=60, B9=60, SCL=20)
        files = {b: one(path.glob(f'GRANULE/*/IMG_DATA/R{res}m/*_'
                                 f'{b if b == "SCL" or b == "B8A" else "B" + b[1:].zfill(2)}'
                                 f'_{res}m.jp2'), b) for b, res in resolutions.items()}
        orbit = None
    elif key == 's1':
        match = re.match(r'(S1[AB])_IW_GRDH_1SDV_(\d{8}T\d{6})_\d{8}T\d{6}_(\d{6})_',
                         path.name)
        if not match:
            raise ValueError(f'Expected Sentinel-1 A/B IW GRDH dual VV/VH: {path}')
        mission, acquired, absolute_orbit = match.groups()
        acquired = datetime.strptime(acquired, '%Y%m%dT%H%M%S').isoformat() + 'Z'
        orbit = (int(absolute_orbit) - {'S1A': 73, 'S1B': 27}[mission]) % 175 + 1
        files = {b: path / f'Sigma0_{b}_db.tif' for b in BANDS['s1']}
        parameters = dict(units='dB', calibration='sigma0',
                          gee_processing_equivalence='not_yet_validated')
    else:
        raise ValueError(key)
    for file in [*files.values(), *metadata]:
        if not file.is_file():
            raise FileNotFoundError(file)
    with rasterio.open(next(iter(files.values()))) as src:
        if src.crs is None:
            raise ValueError(f'Missing source CRS: {path}')
        bounds = transform_bounds(src.crs, crs, *src.bounds, densify_pts=21)
    snapshot = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in [*files.values(), *metadata]}
    return dict(key=key, asset_id='local:' + str(path.resolve()), time=timestamp(acquired),
                orbit=orbit, bounds=list(bounds), files={b: str(p) for b, p in files.items()},
                parameters=parameters, snapshot=snapshot)


def discover(key, patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'Local source pattern matched no directories: {pattern}')
        for root in matches:
            root = Path(root)
            if key == 's1':
                yield from (p.parent for p in root.rglob('Sigma0_VV_db.tif'))
            elif key == 's2':
                yield from (p.parent for p in root.rglob('MTD_MSIL2A.xml'))
            else:
                yield from (p.parent for p in root.rglob('*_MTL.json'))


def catalog(root, plan, settings):
    destination = Path(root) / 'local_catalog.json'
    if settings.get('s1_units') != 'sigma0_db':
        raise ValueError('Local configuration must declare S1 sigma0_db units')
    contract = dict(version=VERSION, plan=fingerprint(plan), settings=settings)
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous['contract'] != contract:
            raise ValueError('Local catalog configuration changed; use a new root')
        return previous
    height, width = input_shape(plan['grid'])
    tr = input_transform(plan['grid'])
    area = box(tr.c, tr.f - height * 10, tr.c + width * 10, tr.f)
    start = timestamp(regional_slots(plan['periods'][0]['time'], plan['config'])[0][0])
    end = timestamp(regional_slots(plan['periods'][-1]['time'], plan['config'])[-1][1])
    scenes, seen, excluded = [], set(), []
    progress = dict(status='indexing', sensors={})
    progress_path = Path(root) / 'local_catalog_progress.json'
    with closing(CatalogCheckpoint(Path(root) / 'local_catalog.sqlite', contract)) as checkpoint:
        for key in BANDS:
            LOG.info('Discovering local %s sources (reuse saved file list when available)', key)
            progress['sensor'] = key
            atomic_json(progress_path, progress)
            paths = checkpoint.candidates(key, lambda: discover(key, settings['roots'][key]))
            counts = dict(candidates=len(paths), processed=0, inspected=0, resumed=0,
                          accepted=0, excluded=0, outside_time=0, outside_region=0)
            progress['sensors'][key] = counts
            atomic_json(progress_path, progress)
            for index, path in enumerate(paths):
                try:
                    # Skip unrelated years before opening raster/XML files on shared storage.
                    match = re.search(r'(20\d{6})(?:T|_)', path.name)
                    if match:
                        day = datetime.strptime(match[1], '%Y%m%d').replace(tzinfo=UTC)
                        if not start - 86400 <= int(day.timestamp()) < end:
                            counts['outside_time'] += 1
                            counts['processed'] = index + 1
                            continue
                    result = checkpoint.get(key, path)
                    if result is None:
                        try:
                            result = dict(scene=inspect_scene(key, path, plan['grid']['crs']))
                        except ExcludedScene as exc:
                            result = dict(reason=exc.reason, snapshot=exc.snapshot)
                        checkpoint.save(key, path, result)
                        counts['inspected'] += 1
                    else:
                        counts['resumed'] += 1
                    if 'scene' not in result:
                        excluded.append(dict(key=key, path=str(path), reason=result['reason']))
                        counts['excluded'] += 1
                    else:
                        scene = result['scene']
                        if not start <= scene['time'] < end:
                            counts['outside_time'] += 1
                        elif not box(*scene['bounds']).intersects(area):
                            counts['outside_region'] += 1
                        elif scene['asset_id'] not in seen:
                            scenes.append(scene)
                            seen.add(scene['asset_id'])
                            counts['accepted'] += 1
                    counts['processed'] = index + 1
                    if (index + 1) % 100 == 0:
                        atomic_json(progress_path, progress)
                        LOG.info('Indexed %s %d/%d; resumed=%d, excluded=%d', key,
                                 index + 1, len(paths), counts['resumed'], counts['excluded'])
                except Exception as exc:
                    progress.update(status='failed', path=str(path), error=str(exc))
                    atomic_json(progress_path, progress)
                    raise
            atomic_json(progress_path, progress)
            LOG.info('Local %s: %d regional scenes; excluded=%d, resumed=%d',
                     key, counts['accepted'], counts['excluded'], counts['resumed'])
    scenes.sort(key=lambda s: (s['time'], s['asset_id']), reverse=True)
    result = dict(contract=contract, scenes=scenes)
    if excluded:
        result['excluded'] = excluded
    atomic_json(destination, result)
    progress['status'] = 'complete'
    atomic_json(progress_path, progress)
    return result
