import pytest

from hydrostate.normalization import SUPPORT_VERSION, LEGACY_TASK_NAMES as TASKS


@pytest.fixture
def target_statistics():
    return dict(
        version=1,
        split="train",
        ddof=0,
        support_version=SUPPORT_VERSION,
        training_fingerprint="0" * 64,
        tasks={
            task: dict(count=3, mean=mean, std=std)
            for task, mean, std in zip(TASKS, (0.3, 0.2, 2.0), (0.4, 0.1, 1.5))
        },
    )
