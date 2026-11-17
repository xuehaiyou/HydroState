"""Matched few-sample training budget, independent of the selected architecture."""

import json
from pathlib import Path


def configure_search(cfg, root, *, work_dir, epochs=6, batch_size=4, accumulation=1,
                     train_fraction=.1, seed=None):
    root = Path(root)
    if train_fraction not in (.1, .3):
        raise ValueError('Use an audited nested training fraction: 0.1 or 0.3')
    stats = root / f'stats_{round(train_fraction * 100)}'
    success = json.loads((stats / '_SUCCESS.json').read_text())
    selection = root / 'selection_v2'
    if (Path(success['selection_root']).resolve() != selection.resolve()
            or success['fraction'] != train_fraction):
        raise ValueError('Training statistics belong to a different subset')
    if epochs < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError('Invalid screening budget')
    cfg.model.target_statistics = str(stats / 'target_stats.json')
    cfg.model.site_supervision.statistics = str(stats / 'site_target_stats.json')
    for key in ('train_dataloader', 'val_dataloader'):
        cfg[key].dataset.update(selection_root=str(selection), train_fraction=train_fraction,
                                 validation_fraction=.2)
        cfg[key].batch_size = batch_size
        cfg[key].num_workers = 2
    for key in ('val_evaluator', 'test_evaluator'):
        cfg[key].site_statistics = str(stats / 'site_target_stats.json')
    cfg.model.encoder.gradient_checkpointing = False
    cfg.optim_wrapper.accumulative_counts = accumulation
    cfg.train_cfg.update(max_epochs=epochs, val_interval=2 if epochs % 2 == 0 else 1)
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=.01, by_epoch=False, begin=0,
             end=100 * accumulation),
        dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=epochs,
             T_max=epochs, eta_min_ratio=.01),
    ]
    cfg.load_from = None
    cfg.resume = False
    if seed is not None:
        cfg.randomness.update(seed=seed)
    cfg.work_dir = str(work_dir)
    return cfg
