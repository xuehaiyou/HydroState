import json

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
