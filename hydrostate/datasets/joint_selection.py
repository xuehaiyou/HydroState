"""Apply audited architecture-search selections after full source validation."""

import json
from pathlib import Path

import pandas as pd

from hydrostate.data.site.archives import sha256


def selection_path(root, audit, split, fraction, component):
    record = audit['splits'][split][str(component)]['subsets'][str(float(fraction))]
    path = root / record['manifest']
    if sha256(path) != record['sha256']:
        raise ValueError('Architecture selection manifest changed')
    return path


def apply_selection(dataset, root, train_fraction, validation_fraction):
    root = Path(root)
    audit = json.loads((root / 'audit.json').read_text())
    if audit.get('version') != 1 or dataset.split not in ('train', 'validation'):
        raise ValueError('Search selections support train/development validation only')
    fraction = train_fraction if dataset.split == 'train' else validation_fraction
    training_manifests = []
    for index, component in enumerate(dataset.components):
        path = selection_path(root, audit, dataset.split, fraction, index)
        selected = pd.read_parquet(path)
        original = component.table.set_index('sample_id')
        if (selected.empty or selected.sample_id.duplicated().any()
                or not selected.split.eq(dataset.split).all()
                or not selected.sample_id.isin(original.index).all()):
            raise ValueError('Search selection changes sample identity or split')
        columns = component.table.columns.tolist()
        expected = original.loc[selected.sample_id].reset_index()[columns]
        if not selected[columns].reset_index(drop=True).equals(expected):
            raise ValueError('Search selection modifies immutable sample metadata')
        component.table = selected.reset_index(drop=True)
        component.manifest = str(path)
        training_manifests.append(str(selection_path(root, audit, 'train', train_fraction, index)))
    dataset.manifest = dataset.product.manifest
    dataset.training_manifests = training_manifests
    dataset.offsets = [0]
    for component in dataset.components:
        dataset.offsets.append(dataset.offsets[-1] + len(component))
