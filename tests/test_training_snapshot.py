import pandas as pd
import pytest

from tools.data.prepare_training_snapshot import select_samples


def sample_table(count=120):
    return pd.DataFrame({
        "sample_id": [f"sample_{i:04d}" for i in range(count)],
        "split": [["train", "train", "train", "validation", "test"][i % 5] for i in range(count)],
        "frequency_bin": [i % 7 for i in range(count)],
    })


def test_snapshot_is_independent_of_source_order_and_preserves_splits():
    table = sample_table()
    first = select_samples(table, 40)
    second = select_samples(table.sample(frac=1, random_state=3), 40)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.split) == {"train", "validation", "test"}
    assert first.set_index("sample_id").split.equals(table.set_index("sample_id").loc[first.sample_id].split)


def test_extension_retains_every_previous_sample_after_new_data_arrive():
    previous = select_samples(sample_table(80), 40)
    larger = select_samples(sample_table(120), 70, previous)
    assert len(larger) == 70
    assert set(previous.sample_id) <= set(larger.sample_id)
    pd.testing.assert_frame_equal(
        previous, larger[larger.sample_id.isin(previous.sample_id)].reset_index(drop=True)
    )


def test_extension_rejects_changed_records_and_shrinking():
    table = sample_table()
    previous = select_samples(table, 40)
    changed = table.copy()
    changed.loc[changed.sample_id == previous.iloc[0].sample_id, "split"] = "changed"
    with pytest.raises(ValueError, match="records changed"):
        select_samples(changed, 70, previous)
    with pytest.raises(ValueError, match="cannot shrink"):
        select_samples(table, 20, previous)
