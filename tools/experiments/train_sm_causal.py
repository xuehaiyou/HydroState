#!/usr/bin/env python
"""Train one frozen run, checking prerequisite and provenance before any update."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
import hydrostate.experiments.sm_causal  # noqa
from hydrostate.experiments.sm_causal import CausalDataset
from hydrostate.registry import DATASETS
from hydrostate.cli import configure_target_statistics


@DATASETS.register_module()
class CausalSmokeDataset(CausalDataset):
    def __init__(self, limit=4, **kwargs):
        super().__init__(**kwargs)
        self.table = self.table.iloc[:limit].reset_index(drop=True)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@HOOKS.register_module()
class CausalProvenanceHook(Hook):
    def before_train(self, runner):
        if runner.rank:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        hashes = {}
        for group, module in [('encoder',model.encoder),('decoder',model.decoder),('sm_head',model.heads['soil_moisture'])]:
            h = hashlib.sha256()
            for name,p in module.state_dict().items():
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            hashes[group] = h.hexdigest()
        report = dict(initialization_sha256=hashes,world_size=runner.world_size,
            active_tasks=model.active_tasks,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        (Path(runner.work_dir)/'initialization.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    cfg = Config.fromfile(args.config)
    protocol = cfg.causal_protocol
    root = Path(protocol.experiment_root)
    expected = json.loads((root/'config_sha256.json').read_text()).get(str(args.config.resolve()))
    if expected != sha(args.config):
        raise RuntimeError('Configuration changed after submission snapshot')
    gate = json.loads((root/'gate/passed.json').read_text())
    if not gate['passed'] or gate['component_sha256'] != sha('hydrostate/experiments/sm_causal.py'):
        raise RuntimeError('Missing or stale GPU consistency gate')
    if int(os.environ.get('WORLD_SIZE','1')) != protocol.world_size:
        raise RuntimeError('Formal experiment requires two DDP ranks, batch 16 per rank')
    if sha(cfg.train_dataloader.dataset.manifest) != protocol.manifest_sha256:
        raise RuntimeError('Manifest changed after preparation')
    if sha(cfg.train_dataloader.sampler.schedule) != protocol.schedule_sha256:
        raise RuntimeError('Batch schedule changed after preparation')
    if protocol.temporal_variant != 'original':
        # Never silently launch temporal runs before checking their changed support.
        approval = root/'temporal_support_passed.json'
        if not approval.exists():
            raise RuntimeError('Temporal input support audit must run before C experiments')
        audit = json.loads(approval.read_text())
        if (not audit['passed'] or audit['component_sha256'] != gate['component_sha256']
                or audit['manifest_sha256'].get(cfg.train_dataloader.dataset.manifest) != protocol.manifest_sha256):
            raise RuntimeError('Temporal support audit is stale or belongs to another manifest')
    if args.smoke:
        cfg.work_dir = str(root/'smoke'/protocol.run_id)
        cfg.train_cfg = dict(type='IterBasedTrainLoop',max_iters=2,val_interval=2)
        cfg.val_dataloader.dataset.split = 'validation'
        cfg.val_dataloader.dataset = dict(cfg.val_dataloader.dataset)
        cfg.val_dataloader.dataset['type'] = 'CausalSmokeDataset'
        # Limit validation without touching training schedules or target statistics.
        cfg.val_dataloader.dataset['limit'] = 4
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook',by_epoch=False,interval=2,max_keep_ckpts=1)
        cfg.default_hooks.logger.interval = 1
    elif not (root/'smoke/passed.json').exists():
        raise RuntimeError('Two-GPU smoke has not completed')
    cfg.launcher = 'pytorch'
    cfg.custom_hooks = [dict(type='CausalProvenanceHook')]
    configure_target_statistics(cfg,training=True)
    runner = Runner.from_cfg(cfg)
    runner.train()
    if runner.rank == 0:
        output = dict(config=str(args.config.resolve()),config_sha256=sha(args.config),
            smoke=args.smoke,iterations=runner.iter,component_sha256=gate['component_sha256'])
        (Path(runner.work_dir)/'completed.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
