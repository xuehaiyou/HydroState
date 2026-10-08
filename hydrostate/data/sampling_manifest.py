"""Cross-run identities and geographic constraints for GEE sample manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def geographic_block(row):
    return f"{int((row['lon'] + 180) // 5)}:{int((row['lat'] + 90) // 5)}"


def geographic_split(row, seed):
    # Preserve the original gee_sampling.fingerprint([seed, block]) contract.
    payload = json.dumps([seed, geographic_block(row)], sort_keys=True).encode()
    value = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'validation' if value < 85 else 'test'


def location_key(row):
    transform = row['transform']
    if isinstance(transform, str):
        transform = json.loads(transform)
    return (row['crs'], *map(float, transform))


def sample_key(row):
    """Physical window/date identity independent of split and sample_id."""
    return (*location_key(row), str(row['target_time'])[:10])


def read_manifest(path, split_seed):
    table = pd.read_parquet(path)
    required = {'sample_id', 'lon', 'lat', 'crs', 'transform', 'target_time', 'split',
                'input_shard', 'input_row', 'label_shard', 'label_row'}
    if missing := required.difference(table.columns):
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    for row in table.to_dict('records'):
        if row['split'] != geographic_split(row, split_seed):
            raise ValueError(f"{path}: geographic split mismatch for {row['sample_id']}; "
                             f'use the original split seed ({split_seed} requested)')
    return table


def source_contract(paths):
    """Pin exclusions to exact completed manifests, including across resume."""
    return [{'path': str(Path(p).resolve()),
             'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]


class SamplingConstraints:
    def __init__(self, rows=()):
        self.keys = set()
        self.location_counts = {}
        self.occupied = {}
        for row in rows:
            self.add(row)

    def add(self, row):
        key = sample_key(row)
        if key in self.keys:
            return
        self.keys.add(key)
        location = location_key(row)
        self.location_counts[location] = self.location_counts.get(location, 0) + 1
        # Affine tuple is a,b,c,d,e,f: translation at indices 3 and 6 with CRS prefix.
        month = str(row['target_time'])[:7]
        self.occupied.setdefault((month, row['crs']), []).append((location[3], location[6]))

    def allows(self, row, month, max_dates):
        location = location_key(row)
        if self.location_counts.get(location, 0) >= max_dates:
            return False
        return not any(abs(location[3] - x) < 1200 and abs(location[6] - y) < 1200
                       for x, y in self.occupied.get((month, row['crs']), []))
