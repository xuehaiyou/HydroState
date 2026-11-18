import json
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from hydrostate.experiments.expert_search import EFFECTIVE_BATCH, VARIANTS, prepare, submit_once


def test_new_plan_freezes_development_only_reload_and_station_selection(tmp_path, monkeypatch):
    selection = tmp_path / 'selection'
    stats = selection / 'stats_10'
    stats.mkdir(parents=True)
    (stats / '_SUCCESS.json').write_text(json.dumps(dict(
        selection_root=str(selection / 'selection_v2'), fraction=.1)))
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'abcdef\n')
    root = tmp_path / 'runs'
    prepare(root, selection)
    assert len(VARIANTS) == 8
    for variant in VARIANTS:
        cfg = Config.fromfile(str(root / variant / 'base.py'))
        assert cfg.val_dataloader == cfg.test_dataloader
        assert cfg.val_dataloader.dataset.split == 'validation'
        assert cfg.val_evaluator == cfg.test_evaluator
        assert cfg.val_evaluator.selection == 'site'
        assert cfg.default_hooks.checkpoint.save_best == 'site_score'
        assert (cfg.train_dataloader.batch_size * cfg.optim_wrapper.accumulative_counts
                == EFFECTIVE_BATCH)
        assert cfg.load_from is None and not cfg.resume
    with pytest.raises(FileExistsError):
        prepare(root, selection)


def test_submit_receipt_prevents_duplicates_and_ambiguous_retry(tmp_path, monkeypatch):
    calls = []

    def submit(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='321\n', stderr='', returncode=0,
                               check_returncode=lambda: None)

    monkeypatch.setattr('subprocess.run', submit)
    record = tmp_path / 'submission.json'
    assert submit_once(record, ['sbatch', 'a.sh'], {}) == '321'
    assert submit_once(record, ['sbatch', 'a.sh'], {}) == '321'
    assert len(calls) == 1
    record.write_text(json.dumps(dict(status='submitting')))
    with pytest.raises(RuntimeError, match='prior submission'):
        submit_once(record, ['sbatch', 'a.sh'], {})
