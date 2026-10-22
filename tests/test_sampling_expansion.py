"""Offline checks for expanding immutable GEE Zarr datasets."""
import json
from datetime import UTC

import pandas as pd
import pytest
import zarr

from hydrostate.legacy.sampling import fingerprint
from hydrostate.data.merge_samples import COMPATIBILITY, merge, validate_shards
from hydrostate.data.sampling_manifest import (
    SamplingConstraints,
    geographic_block,
    geographic_split,
    read_manifest,
    sample_key,
    source_contract,
)
from hydrostate.legacy.schema import LegacySchema as Schema


def sample(sid='a', x=0, day='2020-01-01'):
    row = dict(sample_id=sid, lon=12.5, lat=22.5, crs='EPSG:32633',
               transform=json.dumps([10, 0, x, 0, -10, 0]), target_time=day,
               input_shard=f'zarr/{sid}/inputs.zarr', input_row=0,
               label_shard=f'zarr/{sid}/labels.zarr', label_row=0,
               frequency_bin=0, climate_class=1)
    row['split'] = geographic_split(row, 42)
    return row


def write_run(root, rows):
    root.mkdir(parents=True)
    for row in rows:
        for column, specs in [('input_shard', Schema().inputs),
                              ('label_shard', Schema().targets)]:
            path = root / row[column]
            group = zarr.open_group(str(path), mode='w')
            attrs = {key: 'fixture' for key in COMPATIBILITY}
            attrs.update(schema_version='3.0', sample=row)
            group.attrs.update(attrs)
            for name, spec in specs.items():
                group.create_dataset(name, shape=(1, *spec.tail_shape), dtype=spec.dtype)
            if column == 'label_shard':
                native = group.require_group('et_native/0')
                for name, shape in [('weights', (1, 40, 40)), ('value', (1,)), ('valid', (1,))]:
                    native.create_dataset(name, shape=shape, dtype='float32')
            (path / '_SUCCESS.json').write_text('{"complete": true}')
    path = root / 'samples.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_split_preserves_original_hash():
    for lon in (-179, -20, 12.5, 170):
        row = dict(lon=lon, lat=22.5)
        h = int(fingerprint([42, geographic_block(row)])[:8], 16) % 100
        expected = 'train' if h < 70 else 'validation' if h < 85 else 'test'
        assert geographic_split(row, 42) == expected


def test_constraints_old_and_new_dates_and_spacing():
    old = sample()
    constraints = SamplingConstraints([old, old])
    assert not constraints.allows(sample(x=1190), '2020-01', 4)
    assert constraints.allows(sample(x=1200), '2020-01', 4)
    assert constraints.allows(sample(), '2020-02', 4)
    for month in (2, 3, 4):
        constraints.add(sample(day=f'2020-{month:02}-01'))
    assert not constraints.allows(sample(), '2020-05', 4)
    changed = dict(old, sample_id='different', split='test')
    assert sample_key(changed) == sample_key(old)


def test_merge_keeps_zarr_and_rebases_paths_and_deduplicates(tmp_path):
    old = write_run(tmp_path / 'old', [sample()])
    new = write_run(tmp_path / 'new', [sample('dup'), sample('b', x=2400)])
    before = (tmp_path / 'old/zarr/a/inputs.zarr/.zattrs').read_bytes()
    out = tmp_path / 'merged'
    report = merge([old, new], out)
    table = pd.read_parquet(out / 'samples.parquet')
    assert table.sample_id.tolist() == ['a', 'b']
    assert report['duplicates_removed'] == 1
    assert all(__import__('pathlib').Path(p).is_absolute() for p in table.input_shard)
    assert (tmp_path / 'old/zarr/a/inputs.zarr/.zattrs').read_bytes() == before
    assert not (out / 'zarr').exists()
    with pytest.raises(FileExistsError):
        merge([old, new], out)


def test_merge_rejects_cross_run_spacing(tmp_path):
    old = write_run(tmp_path / 'old', [sample()])
    new = write_run(tmp_path / 'new', [sample('b', x=1000)])
    with pytest.raises(ValueError, match='spacing/date-count'):
        merge([old, new], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_source_changes_are_pinned_and_split_conflicts_fail(tmp_path):
    source = write_run(tmp_path / 'old', [sample()])
    before = source_contract([source])
    table = pd.read_parquet(source)
    table.loc[0, 'split'] = 'test' if table.loc[0, 'split'] != 'test' else 'train'
    table.to_parquet(source)
    assert source_contract([source]) != before
    with pytest.raises(ValueError, match='geographic split mismatch'):
        read_manifest(source, 42)


def test_merge_rejects_preprocessing_changes(tmp_path):
    old = write_run(tmp_path / 'old', [sample()])
    new = write_run(tmp_path / 'new', [sample('b', x=2400)])
    group = zarr.open_group(str(tmp_path / 'new/zarr/b/inputs.zarr'), mode='a')
    group.attrs['normalization_hash'] = 'changed'
    with pytest.raises(ValueError, match='incompatible preprocessing'):
        merge([old, new], tmp_path / 'out')


def test_incomplete_zarr_rejected(tmp_path):
    row = sample()
    write_run(tmp_path / 'old', [row])
    (tmp_path / 'old' / row['label_shard'] / '_SUCCESS.json').unlink()
    with pytest.raises(ValueError, match='Incomplete shard'):
        validate_shards(row, tmp_path / 'old', {})


def test_copy_merge_is_self_contained_and_preserves_zarr_bytes(tmp_path):
    old = write_run(tmp_path / 'old', [sample()])
    new = write_run(tmp_path / 'new', [sample('b', x=2400)])
    out = tmp_path / 'merged'
    merge([old, new], out, copy_zarr=True)
    table = pd.read_parquet(out / 'samples.parquet')
    for source, row in zip([old, new], table.to_dict('records'), strict=True):
        for column in ('input_shard', 'label_shard'):
            relative = __import__('pathlib').Path(row[column])
            assert not relative.is_absolute()
            for original in (source.parent / relative).rglob('*'):
                if original.is_file():
                    copied = out / relative / original.relative_to(source.parent / relative)
                    assert original.read_bytes() == copied.read_bytes()
        validate_shards(row, out, {})


def test_planner_excludes_old_windows_and_keeps_split_seed(tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import hydrostate.legacy.sampling as sampling

    old_row = sample()
    old_row['crs'] = 'EPSG:32633'
    old = write_run(tmp_path / 'old', [old_row])
    root = tmp_path / 'new'
    root.mkdir()
    args = SimpleNamespace(exclude_manifest=[old], split_seed=42, seed=43,
                           climate_asset='climate', climate_band=None, start_year=2020,
                           end_year=2020, samples=1, max_dates_per_location=4,
                           min_valid=.5, latest_days=3, sm_days=3, allow_shortfall=False)
    ee = MagicMock()
    ee.Image.return_value.getInfo.return_value = {'bands': [{'id': 'climate'}]}
    features = [{'properties': {'stratum': 10}, 'priority': i,
                 'geometry': {'coordinates': [lon, 22.5]}}
                for i, lon in enumerate([12.5, 13.5])]
    monkeypatch.setattr(sampling, 'candidate_pool', lambda *a: (features, {10: 100}))
    monkeypatch.setattr(sampling, 'quota',
                        lambda n, keys: {0: 1} if keys == sampling.FREQUENCY_BINS
                        else {'2020-01': 1})
    monkeypatch.setattr(sampling, 'climate_quota', lambda *a: {10: 1})
    monkeypatch.setattr(sampling, 'grid', lambda lon, lat:
                        ('EPSG:32633', [10, 0, 0 if lon == 12.5 else 2400, 0, -10, 0]))
    calls = []
    def select(ee, key, row, *a):
        calls.append(row['lon'])
        if key in ('sm', 'pml'):
            return None
        return {'system:time_start': int(datetime(2020, 1, 15, tzinfo=UTC)
                                         .timestamp() * 1000), 'asset_id': key}
    monkeypatch.setattr(sampling, 'select_scene', select)
    rows = sampling.plan(ee, args, root)
    assert len(rows) == 1 and rows[0]['lon'] == 13.5
    assert set(calls) == {13.5}  # Old window excluded before making scene requests.
    assert rows[0]['split'] == geographic_split(rows[0], 42)
    (root / 'sample_index.parquet').unlink()
    calls.clear()
    resumed = sampling.plan(ee, args, root)
    for record in [rows[0], resumed[0]]:
        for key in ('inputs', 'labels', 'transform'):
            record[key] = json.loads(record[key])
    assert resumed == rows  # Recover the cached accepted selection.
    assert not calls
