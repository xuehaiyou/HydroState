import json

import pytest
from mmengine.config import Config

from hydrostate.experiments.search_recipe import configure_search


def test_controls_share_exposure_and_never_load_task_weights(tmp_path):
    (tmp_path / 'stats_10').mkdir()
    (tmp_path / 'stats_10/_SUCCESS.json').write_text(json.dumps(dict(
        selection_root=str(tmp_path / 'selection_v2'), fraction=.1)))
    controls = [configure_search(Config.fromfile(f'configs/{name}.py'), tmp_path,
                                  work_dir=tmp_path / name)
                for name in ('hydrostate_joint_shared', 'hydrostate_joint_independent')]
    a, b = controls
    assert a.model.type != b.model.type
    for key in ('train_dataloader', 'val_dataloader', 'optim_wrapper', 'param_scheduler',
                'train_cfg', 'randomness'):
        assert a[key] == b[key]
    assert a.model.target_statistics == b.model.target_statistics
    assert a.load_from is b.load_from is None
    assert not a.resume and not b.resume
    assert a.train_cfg.max_epochs == 6 and a.train_cfg.val_interval == 2
    assert a.train_dataloader.batch_size == 4


def test_confirmation_uses_own_training_scales_and_fixed_development_set(tmp_path):
    stats = tmp_path / 'stats_30'
    stats.mkdir()
    marker = stats / '_SUCCESS.json'
    marker.write_text(json.dumps(dict(
        selection_root=str(tmp_path / 'selection_v2'), fraction=.3)))
    cfg = configure_search(Config.fromfile('configs/hydrostate_joint_independent.py'),
                           tmp_path, work_dir=tmp_path / 'confirm', epochs=12,
                           train_fraction=.3, seed=43)
    assert cfg.train_dataloader.dataset.train_fraction == .3
    assert cfg.val_dataloader.dataset.validation_fraction == .2
    assert cfg.model.target_statistics == str(stats / 'target_stats.json')
    assert cfg.model.site_supervision.statistics == str(stats / 'site_target_stats.json')
    assert cfg.val_evaluator.site_statistics == cfg.model.site_supervision.statistics
    assert cfg.randomness.seed == 43
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.load_from is None and not cfg.resume
    marker.write_text(json.dumps(dict(
        selection_root=str(tmp_path / 'selection_v2'), fraction=.1)))
    with pytest.raises(ValueError, match='different subset'):
        configure_search(cfg, tmp_path, work_dir=tmp_path / 'bad', train_fraction=.3)


def test_station_selection_changes_checkpoint_without_changing_training_loss(tmp_path):
    (tmp_path / 'stats_10').mkdir()
    (tmp_path / 'stats_10/_SUCCESS.json').write_text(json.dumps(dict(
        selection_root=str(tmp_path / 'selection_v2'), fraction=.1)))
    cfg = configure_search(Config.fromfile('configs/hydrostate_joint_ple.py'), tmp_path,
                           work_dir=tmp_path / 'ple', station_selection=True)
    assert cfg.default_hooks.checkpoint.save_best == 'site_score'
    assert cfg.default_hooks.checkpoint.rule == 'greater'
    assert cfg.val_evaluator.selection == cfg.test_evaluator.selection == 'site'
    assert cfg.model.site_supervision.loss_weights == dict(site_sm=2., site_et=2.)
