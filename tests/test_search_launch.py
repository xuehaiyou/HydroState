import json
import math
from types import SimpleNamespace

import pytest

from hydrostate.experiments.search_launch import estimate_budget, launch


def test_budget_uses_all_train_windows_and_scheduled_development_evaluations():
    common = dict(status='passed', peak_allocated_gib=20., peak_external_gib=1.,
                  gpu_total_gib=44., memory_samples=10)
    bench = dict(common, batch_size=4, train_seconds_per_sample=.4,
                 validation_seconds_per_sample=.2, train_samples=4528, validation_samples=2261)
    smoke = dict(common, micro_batch=4, checkpoint_parameters_verified=10,
                 elapsed_seconds=80., selection={k: dict(samples=20)
                                                for k in ('train', 'val', 'test')})
    result = estimate_budget(bench, smoke)
    expected = 6 * (4528 * .4 + 64) + 3 * 2261 * .2
    assert result['estimated_seconds'] == expected
    assert result['seconds'] == math.ceil(expected * 1.1 / 60) * 60
    smoke['status'] = 'oom'
    with pytest.raises(ValueError, match='acceptance'):
        estimate_budget(bench, smoke)


def test_sequential_submissions_record_once_and_never_test(tmp_path, monkeypatch):
    from mmengine.config import Config

    root, debug = tmp_path, tmp_path / 'debug'
    debug.mkdir()
    release = root / 'releases/abc'
    release.mkdir(parents=True)
    stats = root / 'stats_10'
    stats.mkdir()
    (stats / '_SUCCESS.json').write_text(json.dumps(dict(
        selection_root=str(root / 'selection_v2'), fraction=.1)))
    (root / 'screening_environment.json').write_text(json.dumps(dict(
        HYDROSTATE_CODE_ROOT=str(release), PYTHONPATH=str(release))))
    (debug / 'receipt.json').write_text(json.dumps(dict(stdout='123\n', commit='abc')))
    variants = dict(A='shared', B='independent', CGC='cgc', MLoRE='mlore')
    (debug / 'passed.json').write_text(json.dumps(dict(variants=list(variants), batch_size=4)))
    common = dict(status='passed', peak_allocated_gib=20., peak_external_gib=1.,
                  gpu_total_gib=44., memory_samples=10)
    for variant, base in variants.items():
        Config.fromfile(f'configs/hydrostate_joint_{base}.py').dump(str(root/f'{variant}_10.py'))
        (debug / f'{variant}_4.json').write_text(json.dumps(dict(
            common, gpu='NVIDIA L40S', batch_size=4, train_seconds_per_sample=.4,
            validation_seconds_per_sample=.2, train_samples=4528, validation_samples=2261)))
        smoke = debug / f'smoke_{variant}'
        smoke.mkdir()
        (smoke / 'result.json').write_text(json.dumps(dict(
            common, micro_batch=4, checkpoint_parameters_verified=10, elapsed_seconds=80.,
            selection={k: dict(samples=20) for k in ('train', 'val', 'test')})))
    calls = []

    def submit(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=f'{1000+len(calls)}\n', stderr='', returncode=0,
                               check_returncode=lambda: None)

    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '123|COMPLETED|0:0|\n')
    monkeypatch.setattr('subprocess.run', submit)
    plan = launch(root, debug, submit=True)
    assert len(calls) == 4
    assert not any('dependency' in arg for arg in calls[0])
    for index in range(1, 4):
        assert f'--dependency=afterok:{1000+index}' in calls[index]
    assert launch(root, debug, submit=True) == plan
    assert len(calls) == 4
    for variant in variants:
        directory = root / 'screen_10' / variant
        script = (directory / 'run.sh').read_text()
        assert 'tools/train.py' in script and 'tools/test.py' not in script
        cfg = Config.fromfile(str(directory / 'config.py'))
        assert cfg.load_from is None and not cfg.resume
        assert cfg.train_cfg.max_epochs == 6
