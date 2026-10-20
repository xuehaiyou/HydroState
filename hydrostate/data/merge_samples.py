"""Validate and publish a manifest referencing existing, unchanged GEE Zarr shards."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import zarr

from hydrostate.data.sampling_manifest import (
    SamplingConstraints,
    geographic_block,
    read_manifest,
    sample_key,
    source_contract,
)
from hydrostate.data.temporal_sampling import (
    LEGACY_SAMPLE_CONFIG,
    sample_row_config,
    sample_temporal_config,
)
from hydrostate.storage import Schema

LOG = logging.getLogger(__name__)
COMPATIBILITY = ('schema_version', 'normalization', 'normalization_hash',
                 'olmoearth_pretrain_minimal', 'resolution', 'water_encoding',
                 'coarse_labels', 'et_temporal_support')


def validate_shards(row, root, contract):
    schema = Schema()
    for column, row_column, specs in (
        ('input_shard', 'input_row', schema.inputs),
        ('label_shard', 'label_row', schema.targets),
    ):
        path = Path(row[column])
        path = (path if path.is_absolute() else root / path).resolve()
        marker = path / '_SUCCESS.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('complete'):
            raise ValueError(f'Incomplete shard: {path}')
        group = zarr.open_group(str(path), mode='r')
        attrs = dict(group.attrs)
        if missing := set(COMPATIBILITY).difference(attrs):
            raise ValueError(f'{path}: missing preprocessing metadata {sorted(missing)}')
        current = {key: attrs[key] for key in COMPATIBILITY}
        current["input_temporal_config"] = sample_temporal_config(
            attrs.get("input_temporal_config", LEGACY_SAMPLE_CONFIG)
        )
        if current["input_temporal_config"] != sample_row_config(row):
            raise ValueError(f"{path}: sample temporal metadata differs from manifest")
        if current['schema_version'] != schema.version:
            raise ValueError(f'{path}: incompatible schema version')
        if contract and current != contract:
            raise ValueError(f'{path}: incompatible preprocessing/label metadata')
        contract.update(current)
        stored = attrs.get('sample', {})
        if (stored.get('sample_id') != row['sample_id']
                or stored.get('split') != row['split'] or sample_key(stored) != sample_key(row)):
            raise ValueError(f'{path}: Zarr sample metadata differs from manifest')
        index = int(row[row_column])
        if index < 0 or index != row[row_column]:
            raise ValueError(f'{path}: invalid row index')
        for name, spec in specs.items():
            if name not in group:
                raise ValueError(f'{path}: missing array {name}')
            array = group[name]
            if (array.shape[0] <= index or tuple(array.shape[1:]) != spec.tail_shape
                    or str(array.dtype) != str(spec.dtype)):
                raise ValueError(f'{path}/{name}: incompatible array shape/dtype')
        if column == 'label_shard':
            native = group.get(f'et_native/{index}')
            if native is None or any(k not in native for k in ('weights', 'value', 'valid')):
                raise ValueError(f'{path}: missing native PML supervision')
            n = native['value'].shape[0]
            if native['weights'].shape != (n, 40, 40) or native['valid'].shape != (n,):
                raise ValueError(f'{path}: invalid native PML array shapes')
        row[column] = str(path)


def merge(sources, output, split_seed=42, max_dates=4, copy_zarr=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new version directory: {output}')
    contracts = source_contract(sources)
    rows, seen, ids, metadata = [], set(), {}, {}
    constraints = SamplingConstraints()
    duplicates = 0
    source_counts = {}
    temporal_config = None
    for source in sources:
        source = Path(source).resolve()
        table = read_manifest(source, split_seed)
        source_counts[str(source)] = len(table)
        for row in table.to_dict('records'):
            current_temporal = sample_row_config(row)
            if temporal_config is not None and current_temporal != temporal_config:
                raise ValueError("Cannot merge samples with different temporal sampling contracts")
            temporal_config = current_temporal
            key = sample_key(row)
            sid = row['sample_id']
            if sid in ids and ids[sid] != key:
                raise ValueError(f'sample_id collision: {sid}')
            ids[sid] = key
            if key in seen:
                duplicates += 1
                continue
            if not constraints.allows(row, str(row['target_time'])[:7], max_dates):
                raise ValueError(f'Cross-run spacing/date-count conflict: {sid}')
            validate_shards(row, source.parent, metadata)
            row['source_manifest'] = str(source)
            rows.append(row)
            seen.add(key)
            constraints.add(row)
            if len(rows) % 100 == 0:
                LOG.info('Validated %s unique samples', len(rows))
    if not rows:
        raise ValueError('No completed samples to merge')
    if contracts != source_contract(sources):
        raise RuntimeError('Source manifests changed during merge; retry after sampling completes')
    table = pd.DataFrame(rows)
    report = dict(sources=contracts, source_counts=source_counts, split_seed=split_seed,
                  max_dates_per_location=max_dates, accepted=len(rows),
                  duplicates_removed=duplicates, copied_zarr=copy_zarr,
                  splits=table.groupby('split').size().to_dict(), preprocessing=metadata)
    for column in ('frequency_bin', 'climate_class'):
        if column in table:
            report[column] = (table.groupby(['split', column]).size().rename('count')
                              .reset_index().to_dict('records'))
    report['months'] = (table.assign(month=table.target_time.astype(str).str[:7])
                        .groupby(['split', 'month']).size().rename('count')
                        .reset_index().to_dict('records'))
    report['geographic_blocks'] = len({geographic_block(row) for row in rows})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    if copy_zarr:
        for index, row in enumerate(rows):
            sid = str(row['sample_id'])
            if not sid or sid in ('.', '..') or Path(sid).name != sid:
                raise ValueError(f'Unsafe sample_id for output directory: {sid!r}')
            for column, name in (('input_shard', 'inputs.zarr'), ('label_shard', 'labels.zarr')):
                relative = Path('zarr') / sid / name
                shutil.copytree(row[column], staging / relative)
                table.at[index, column] = str(relative)
            if (index + 1) % 100 == 0:
                LOG.info('Copied %s/%s Zarr pairs', index + 1, len(rows))
        # Recheck all copied schemas before publishing the self-contained directory.
        for row in table.to_dict('records'):
            validate_shards(row, staging, {})
    table.to_parquet(staging / 'samples.parquet', index=False)
    (staging / 'merge_report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    os.rename(staging, output)
    LOG.info('Published %s samples: %s', len(rows), output / 'samples.parquet')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True,
                        help='Completed samples.parquet; repeat in preference order (old first)')
    parser.add_argument('--output', type=Path, required=True, help='New version directory')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--max-dates-per-location', type=int, default=4)
    parser.add_argument('--copy-zarr', action='store_true',
                        help='Copy unchanged Zarr stores into output/zarr; use relative paths')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    merge(args.source, args.output, args.split_seed, args.max_dates_per_location, args.copy_zarr)


if __name__ == '__main__':
    main()
