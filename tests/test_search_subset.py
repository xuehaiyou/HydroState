import pandas as pd
import pytest

from hydrostate.experiments.search_subset import select_nested


def records():
    return pd.DataFrame([dict(sample_id=str(i), split='train', group=str(i % 7),
                              stratum=i % 3, target_time=f'2020-{i % 12 + 1:02}-01',
                              rare=int(i in (2, 7)), common=1) for i in range(90)])


def select(table):
    return select_nested(table, [.1, .3], seed=42, group_column='group',
                         stratum_column='stratum', label_columns=['rare', 'common'],
                         minimum_label_windows=2, preserve_groups=True)


def test_nested_reorder_stable_and_rare_support():
    table = records()
    subsets = select(table)
    shuffled = select(table.sample(frac=1, random_state=9))
    assert set(subsets[.1].sample_id) <= set(subsets[.3].sample_id)
    for fraction, subset in subsets.items():
        assert subset.sample_id.tolist() == shuffled[fraction].sample_id.tolist()
        assert not subset.sample_id.duplicated().any()
        assert subset.rare.sum() == 2
        assert subset.group.nunique() == table.group.nunique()
        assert subset.split.eq('train').all()


def test_reject_mixed_splits_and_duplicate_ids():
    table = records()
    table.loc[0, 'split'] = 'validation'
    with pytest.raises(ValueError, match='split'):
        select(table)
    table = records()
    table.loc[0, 'sample_id'] = table.loc[1, 'sample_id']
    with pytest.raises(ValueError, match='unique'):
        select(table)
