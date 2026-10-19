"""Reuse a frozen AOI without requesting lake geometry over the network."""

import copy
import json
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.input_products import L2_PROFILE
from hydrostate.inference.planning import periods, windows


def initialize_from_template(root, config_path, template_root):
    root, template_root = Path(root), Path(template_root)
    if root.resolve() == template_root.resolve():
        raise ValueError('L2 acquisition requires a new output root')
    source = json.loads((template_root / 'plan.json').read_text())
    config = json.loads(Path(config_path).read_text())
    if config.get('input_profile') != L2_PROFILE:
        raise ValueError('Expected an explicit Landsat L2 SR profile')
    for key in ('crs', 'input_size', 'output_resolution_m', 'core_size', 'stride_size',
                'lake_collection', 'lake_id', 'buffer_m', 'training_manifest'):
        if config.get(key) != source['config'].get(key):
            raise ValueError(f'Template {key} differs; initialize a new AOI instead')
    result = copy.deepcopy(source)
    result['config'] = config
    result['periods'] = periods(config['start'], config['end'])
    if source['identity'].get('pilot'):
        dates = set(config['pilot_dates'])
        result['periods'] = [p for p in result['periods'] if p['time'] in dates]
    result['identity'] = dict(config=config, template_plan_sha256=fingerprint(source),
                              pilot=source['identity'].get('pilot', False))
    result.pop('plan_hash', None)
    result['plan_hash'] = fingerprint(result)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / 'prepare.lock', blocking=False):
        path = root / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != result:
            raise ValueError('Existing plan differs; use a new root')
        atomic_json(path, result)
        import pandas as pd

        pd.DataFrame(windows(result)).to_parquet(root / 'windows.parquet', index=False)
        if (template_root / 'aoi.geojson').exists():
            (root / 'aoi.geojson').write_bytes((template_root / 'aoi.geojson').read_bytes())
    return result
