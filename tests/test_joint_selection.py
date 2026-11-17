import json
from types import SimpleNamespace

import pandas as pd
import pytest

from hydrostate.data.site.archives import sha256
from hydrostate.datasets.joint_selection import apply_selection


class Component(SimpleNamespace):
    def __len__(self):
        return len(self.table)


def fixture(tmp_path):
    table = pd.DataFrame(dict(sample_id=['a', 'b', 'c'], split=['train'] * 3,
                              target_time=['2020-01-01'] * 3, value=[1, 2, 3]))
    selected = table.iloc[[0, 2]].copy()
    path = tmp_path / 'selection.parquet'
    selected.to_parquet(path)
    audit = dict(version=1, splits={'train': {'0': {'subsets': {'0.1': {
        'manifest': path.name, 'sha256': sha256(path)}}}}})
    (tmp_path / 'audit.json').write_text(json.dumps(audit))
    component = Component(table=table, manifest='original')
    dataset = SimpleNamespace(split='train', components=[component], product=component)
    return dataset, path, audit


def test_selection_preserves_values_and_adjusts_size(tmp_path):
    dataset, path, _ = fixture(tmp_path)
    apply_selection(dataset, tmp_path, .1, .2)
    assert dataset.offsets == [0, 2]
    assert dataset.product.table.value.tolist() == [1, 3]
    assert dataset.training_manifests == [str(path)]


def test_reject_changed_metadata_even_if_manifest_hash_updated(tmp_path):
    dataset, path, audit = fixture(tmp_path)
    selected = pd.read_parquet(path)
    selected.loc[selected.index[0], 'value'] = 99
    selected.to_parquet(path)
    audit['splits']['train']['0']['subsets']['0.1']['sha256'] = sha256(path)
    (tmp_path / 'audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='immutable'):
        apply_selection(dataset, tmp_path, .1, .2)
