"""Deterministic, nested metadata-only subsets for architecture screening.

Selection never reads predictions or changes a published split. These manifests
are plans: subset-specific training statistics must be prepared before training.
"""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def diverse_order(table, *, seed, group_column):
    """Visit spatial groups before repeating dates; spread repeats across seasons."""
    if table.sample_id.duplicated().any() or table[group_column].isna().any():
        raise ValueError("Require unique sample IDs and explicit spatial groups")
    frame = table.copy()
    dates = pd.to_datetime(frame.target_time, utc=True)
    frame['_season'] = dates.dt.year.astype(str) + ':' + ((dates.dt.month - 1) // 3).astype(str)
    frame['_hash'] = [hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest()
                      for s in frame.sample_id]
    frame = frame.sort_values('_hash')
    frame['_season_visit'] = frame.groupby([group_column, '_season']).cumcount()
    frame = frame.sort_values(['_season_visit', '_hash'])
    frame['_spatial_visit'] = frame.groupby(group_column).cumcount()
    return frame.sort_values(['_spatial_visit', '_hash']).sample_id.tolist()


def select_nested(table, fractions, *, seed, group_column, stratum_column,
                  label_columns, minimum_label_windows=64, preserve_groups=False):
    """Apply per-stratum quotas plus label floors, retaining nested membership.

    Fractions are nominal: label floors and group preservation may increase size.
    Label counts are manifest counts, not a replacement for supervision-mask audit.
    """
    if table.empty or table.split.nunique() != 1:
        raise ValueError("Select one nonempty published split at a time")
    if table[stratum_column].isna().any():
        raise ValueError("Missing sampling strata")
    if any(not 0 < f <= 1 for f in fractions) or minimum_label_windows < 1:
        raise ValueError("Invalid fraction or supervision floor")
    order = diverse_order(table, seed=seed, group_column=group_column)
    indexed = table.set_index('sample_id').loc[order]
    mandatory = set()
    if preserve_groups:
        mandatory.update(indexed.groupby(group_column, sort=False).head(1).index)
    for label in label_columns:
        if indexed[label].isna().any() or indexed[label].lt(0).any():
            raise ValueError(f'Invalid label counts: {label}')
        mandatory.update(indexed[indexed[label].gt(0)].head(minimum_label_windows).index)
    result = {}
    for fraction in sorted(set(fractions)):
        chosen = set(mandatory)
        for _, group in indexed.groupby(stratum_column, sort=True):
            chosen.update(group.head(math.ceil(len(group) * fraction)).index)
        result[fraction] = indexed.loc[[s for s in order if s in chosen]].reset_index()
    return result


def describe_subset(table, *, group_column, stratum_column, label_columns):
    """Report actual exposure and support; never call label counts accuracy."""
    dates = pd.to_datetime(table.target_time, utc=True)
    return dict(
        windows=len(table), spatial_groups=int(table[group_column].nunique()),
        strata={str(k): int(v) for k, v in table[stratum_column].value_counts().items()},
        years={str(k): int(v) for k, v in dates.dt.year.value_counts().items()},
        months={str(k): int(v) for k, v in dates.dt.month.value_counts().items()},
        labels={label: dict(windows=int(table[label].gt(0).sum()),
                            manifest_valid_count=int(table[label].sum()))
                for label in label_columns},
    )


def prepare(config, output):
    """Validate original data contracts and write selection-only audit artifacts."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    from hydrostate.cli import configure_data_contract
    from hydrostate.registry import DATASETS

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(config)
    configure_data_contract(cfg)
    init_default_scope('hydrostate')
    report = dict(version=1, seed=42, config=str(config), status='selection_only',
                  statistics='Pending subset-only computation; not ready for training', splits={})
    for split, key, fractions in [('train', 'train_dataloader', [.1, .3]),
                                  ('validation', 'val_dataloader', [.2])]:
        dataset = DATASETS.build(dict(cfg[key].dataset))
        parts = {}
        for index, component in enumerate(dataset.components):
            table = component.table.copy()
            table['joint_index'] = range(dataset.offsets[index], dataset.offsets[index + 1])
            table['component'] = index
            table['source_key'] = [f'{index}:{s}' for s in table.sample_id]
            if index == 0:
                labels = ['water_valid', 'snow_ice_valid', 'land_valid',
                          'soil_moisture_valid', 'et_water_native_valid',
                          'et_ice_native_valid', 'et_land_native_valid']
                for label in labels:
                    table[label] = [v[label] for v in table.valid_label_counts]
                group, stratum = 'block', 'sampling_group'
            else:
                labels = ['valid_sm', 'valid_et']
                group, stratum = 'physical_group_id', 'sampling_group'
            subsets = select_nested(table, fractions, seed=42, group_column=group,
                                    stratum_column=stratum, label_columns=labels,
                                    preserve_groups=index > 0)
            part = dict(full=describe_subset(table, group_column=group,
                                            stratum_column=stratum, label_columns=labels),
                        subsets={})
            for fraction, selected in subsets.items():
                filename = f'{split}_{fraction:g}_component{index}.parquet'
                selected.to_parquet(output / filename, index=False)
                part['subsets'][str(fraction)] = dict(
                    manifest=filename,
                    sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                    **describe_subset(selected, group_column=group,
                                      stratum_column=stratum, label_columns=labels))
            parts[str(index)] = part
        report['splits'][split] = parts
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    return report
