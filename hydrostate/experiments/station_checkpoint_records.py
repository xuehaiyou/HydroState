"""Export native station predictions for a development-selected joint checkpoint."""
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

from hydrostate.cli import Runner, configure_target_statistics
from hydrostate.models.site_supervision import station_values


@torch.inference_mode()
def export(config, checkpoint, directory):
    cfg = Config.fromfile(config)
    if cfg.val_dataloader.dataset.split != 'validation':
        raise ValueError('Expected a fixed development validation loader')
    cfg.work_dir = str(directory / 'log')
    configure_target_statistics(cfg, training=True)
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(str(checkpoint), strict=True)
    model = runner.model.eval()
    for split in ('validation', 'test'):
        loader_cfg = deepcopy(cfg.val_dataloader)
        if split == 'test':
            loader_cfg.dataset.split = split
            loader_cfg.dataset.pop('selection_root', None)
        loader = runner.build_dataloader(loader_cfg)
        dataset = loader.dataset
        sites = Subset(dataset, range(dataset.offsets[1], len(dataset)))
        rows = []
        for raw in DataLoader(sites, batch_size=loader.batch_size, collate_fn=loader.collate_fn,
                              num_workers=loader.num_workers):
            batch = model.data_preprocessor(raw, training=False)
            samples = batch['data_samples']
            fields = model._forward(batch['inputs'])
            info = samples['metainfo']
            for task, (pred, target, valid) in station_values(fields, samples).items():
                for i, j in valid.nonzero().tolist():
                    rows.append(dict(task=task, site=info['spatial_group_id'][i],
                                     sample_id=info['sample_id'][i], observation=j,
                                     pred=pred[i,j].item(), target=target[i,j].item()))
        if not rows:
            raise ValueError(f'No station predictions for {split}')
        path = directory / f'{split}.parquet'
        tmp = path.with_suffix('.tmp.parquet')
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'directory'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    export(**vars(args))
