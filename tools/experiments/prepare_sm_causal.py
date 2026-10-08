#!/usr/bin/env python
"""Freeze paired manifests, normalizers, schedules and A/B/C/D run configs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from mmengine.config import Config
from hydrostate.normalization import TASKS, training_fingerprint

BASE = Path('/home/xiaoz/HydroState/outputs/diagnostics_v2_4107932/frozen_epoch32')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def moments(frame, ids, template, manifest):
    """Reuse model-independent target sufficient statistics, not predictions."""
    stats = json.loads(json.dumps(template))
    stats['training_fingerprint'] = training_fingerprint(manifest)
    stats['normalization_source_ids_sha256'] = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    stats['normalization_note'] = 'Targets from original full diagnostic, selected training IDs only; no predictions used.'
    stats['training_samples'] = int((manifest.split == 'train').sum())
    for task in TASKS:
        f = frame[(frame.task == task) & frame.sample_id.isin(ids) & (frame['count'] > 0)]
        n, y, y2 = (float(f[k].sum()) for k in ('count', 'y', 'y2'))
        if n < 2 or y2/n-(y/n)**2 <= 0:
            raise ValueError(f'Insufficient normalization support for {task}')
        stats['tasks'][task] = dict(count=int(n), mean=y/n, std=math.sqrt(y2/n-(y/n)**2),
                                   product_count=int(n), product_samples=len(f), site_count=0)
    return stats


def schedule(pool, sm_ids, steps, seed, require_sm=True):
    rng = np.random.default_rng(seed)
    present = pool.sample_id.isin(sm_ids).to_numpy()
    if require_sm and not present.any():
        raise ValueError('Cannot sample supervised batches without SM labels')
    rows = []
    rejected = 0
    for _ in range(steps):
        while True:
            row = rng.integers(0, len(pool), size=32, dtype=np.int32)
            if not require_sm or present[row].any():
                rows.append(row)
                break
            rejected += 1
    return np.stack(rows), rejected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[3407, 3408, 3409])
    p.add_argument('--steps', type=int, default=2340)
    p.add_argument('--stage', choices=['screen', 'full'], default='screen')
    args = p.parse_args()
    if args.steps < 2 or (args.stage == 'full' and args.steps != 11700):
        raise ValueError('Use 11700 steps for formal full-budget comparisons')
    root = args.out.resolve()
    bundle = root / args.stage
    bundle.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(BASE/'hydrostate_l40s.py')
    manifest_path = Path(cfg.train_dataloader.dataset.manifest)
    original = pd.read_parquet(manifest_path)
    if sha(manifest_path) != cfg.model.target_statistics.manifest_sha256:
        raise ValueError('Source manifest hash differs from frozen diagnostic configuration')
    original = original[original.split.isin(['train', 'validation'])].copy()
    frame = pd.concat([pd.read_csv(BASE/f'job_4172645/latest/{s}_windows.csv')
                       for s in ('train', 'validation')], ignore_index=True)
    if not set(original.sample_id) == set(frame.sample_id):
        raise ValueError('Diagnostic IDs differ from manifest')
    sm = frame[(frame.task == 'soil_moisture') & (frame['count'] > 0)]
    sm_ids = set(sm.sample_id)
    audit = original[['sample_id', 'split', 'lon', 'lat', 'target_time', 'climate_class']].copy()
    audit['block'] = [f'{int((x+180)//5)}:{int((y+90)//5)}' for x,y in zip(audit.lon,audit.lat)]
    audit['sm_valid'] = audit.sample_id.isin(sm_ids)
    labels = original.labels.map(json.loads)
    audit['sm_asset'] = [v['sm']['asset_id'] if v.get('sm') else '' for v in labels]
    audit['sm_offset_days'] = [v['sm'].get('offset_days') if v.get('sm') else None for v in labels]
    audit['native_sm_pixel_id'] = 'unknown: SM projection absent from stored metadata'
    audit['conservative_group'] = audit.block + '/' + audit.sm_asset
    audit['group_note'] = '5-degree block x daily asset, NOT an exact native observation ID'
    targets = sm.set_index('sample_id')
    audit['sm_value'] = audit.sample_id.map(targets['y']/targets['count'])
    audit['modalities'] = audit.sample_id.map(frame.drop_duplicates('sample_id').set_index('sample_id').modalities)
    audit.to_parquet(bundle/'audit.parquet', index=False)
    train_audit = audit[audit.split == 'train']
    blocks = sorted(train_audit.block.unique())
    dev_blocks = set(np.random.default_rng(20260925).permutation(blocks)[:max(1, round(.2*len(blocks)))])
    dev_ids = set(train_audit.loc[train_audit.block.isin(dev_blocks), 'sample_id'])
    pool = original[(original.split == 'train') & ~original.sample_id.isin(dev_ids)].reset_index(drop=True)
    development = original[original.sample_id.isin(dev_ids)].copy()
    development['split'] = 'validation'
    external = original[original.split == 'validation'].copy()
    external['split'] = 'external_validation'
    if not set(development.sample_id) & sm_ids or not set(pool.sample_id) & sm_ids:
        raise ValueError('Inner split lacks SM support')
    if set(audit.loc[audit.split == 'validation','block']) & set(train_audit.block):
        raise ValueError('Original spatial blocks overlap between train and validation')
    write_json(bundle/'split.json', dict(dev_blocks=sorted(dev_blocks),
        training_windows=len(pool), training_sm_windows=int(pool.sample_id.isin(sm_ids).sum()),
        dev_windows=len(development), dev_sm_windows=int(development.sample_id.isin(sm_ids).sum()),
        external_sm_windows=int(external.sample_id.isin(sm_ids).sum()),
        grouping='conservative 5-degree block x daily SM asset; exact native pixel IDs unavailable'))
    manifests, stats = {}, {}
    pools = {'all': pool}
    sm_pool = pool[pool.sample_id.isin(sm_ids)].reset_index(drop=True)
    group_by_id = audit.set_index('sample_id').conservative_group
    # Same nested subsets across paired model seeds: subset seed is explicit.
    # Formal full stage regenerates these for each training seed below.
    subsets = {}
    for seed in args.seeds:
        groups = sorted(set(sm_pool.sample_id.map(group_by_id)))
        order = np.random.default_rng(seed).permutation(groups)
        for fraction in (.25, .5, 1.):
            chosen = set(order[:max(1, math.ceil(len(order)*fraction))])
            key = f'sm{int(fraction*100)}_s{seed}'
            pools[key] = sm_pool[sm_pool.sample_id.map(group_by_id).isin(chosen)].reset_index(drop=True)
            subsets[key] = dict(windows=len(pools[key]), conservative_groups=len(chosen), subset_seed=seed)
    write_json(bundle/'subsets.json', subsets)
    for name, subpool in pools.items():
        table = pd.concat([subpool, development, external], ignore_index=True)
        path = bundle/f'{name}.parquet'
        table.to_parquet(path, index=False)
        manifests[name] = path
        ids = set(subpool.sample_id) if name == 'all' else set(pools[f'sm25_s{name.split("_s")[1]}'].sample_id)
        stats[name] = moments(frame, ids, cfg.model.target_statistics, table)
        stats[name]['source_manifest'] = str(path)
        stats[name]['manifest_sha256'] = sha(path)
        write_json(bundle/f'{name}_statistics.json', stats[name])
    rows = []
    for seed in args.seeds:
        variants = [
            ('A0', 'all', 'nll', TASKS, 'original', 'none', False),
            ('A1', 'all', 'nll', ('soil_moisture',), 'original', 'none', False),
            ('A2', 'all', 'mse', TASKS, 'original', 'none', False),
            ('A3', 'all', 'mse', ('soil_moisture',), 'original', 'none', False),
            *[(f'B{int(f*100)}', f'sm{int(f*100)}_s{seed}', 'mse', ('soil_moisture',), 'original', 'none', False) for f in (.25,.5,1.)],
            ('C0', f'sm100_s{seed}', 'mse', ('soil_moisture',), 'recent', 'none', False),
            ('C1', f'sm100_s{seed}', 'mse', ('soil_moisture',), 'history', 'none', False),
            ('C2', f'sm100_s{seed}', 'mse', ('soil_moisture',), 'history', 'relative', False),
            ('C2control', f'sm100_s{seed}', 'mse', ('soil_moisture',), 'history', 'zero', False),
            ('Dfreeze', f'sm100_s{seed}', 'mse', ('soil_moisture',), 'original', 'none', True),
        ]
        schedules = {}
        for name, subpool in pools.items():
            if name != 'all' and not name.endswith(f'_s{seed}'):
                continue
            indices, rejected = schedule(subpool, sm_ids, args.steps, seed)
            path = bundle/f'{name}_{seed}_batches.npy'
            np.save(path, indices)
            schedules[name] = path
            write_json(path.with_suffix('.json'), dict(seed=seed, steps=args.steps,
                global_batch=32, rejected_unsupervised_batches=rejected,
                unique_windows=int(len(np.unique(indices))), manifest_sha256=sha(manifests[name]),
                note='Common conditioning on at least one global SM observation for all A arms'))
        for variant, name, objective, tasks, temporal, condition, freeze in variants:
            run_id = f'{variant}_s{seed}'
            run = Config(cfg.to_dict())
            run.custom_imports = dict(imports=['hydrostate', 'hydrostate.experiments.sm_causal'], allow_failed_imports=False)
            run.model.type = 'CausalHydroStateModel'
            run.model.active_tasks = tasks
            run.model.freeze_encoder = freeze
            run.model.encoder.type = 'CausalOlmoEarthEncoder'
            run.model.encoder.time_condition = condition
            run.model.loss = dict(type='CausalNLL', objective=objective, min_variance=1e-6)
            run.model.data_preprocessor.type = 'CausalPreprocessor'
            run.model.target_statistics = stats[name]
            run.randomness = dict(seed=seed, deterministic=False)
            run.work_dir = str(root/args.stage/'runs'/run_id)
            for split in ('train','val'):
                loader = run[f'{split}_dataloader']
                loader.dataset.update(type='CausalDataset', manifest=str(manifests[name]), temporal_variant=temporal)
                loader.num_workers = 2
                loader.persistent_workers = True
                loader.batch_size = 16
            run.train_dataloader.sampler = dict(type='FixedGlobalBatchSampler', schedule=str(schedules[name]))
            run.val_evaluator.target_statistics = stats[name]
            # No test loop is built or used by this experiment.
            run.test_dataloader = run.test_evaluator = run.test_cfg = None
            run.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.steps, val_interval=234)
            run.log_processor.by_epoch = False
            run.env_cfg.mp_cfg.mp_start_method = 'spawn'
            run.optim_wrapper.accumulative_counts = 1
            run.param_scheduler = [dict(type='LinearLR',start_factor=.01,by_epoch=False,begin=0,end=125),
                dict(type='CosineAnnealingLR',by_epoch=False,begin=125,end=args.steps,T_max=args.steps-125,eta_min=1e-6)]
            run.default_hooks.checkpoint = dict(type='CheckpointHook',by_epoch=False,interval=234,
                save_best='soil_moisture_rmse',rule='less',max_keep_ckpts=1,save_last=True)
            run.causal_protocol = dict(run_id=run_id, stage=args.stage, experiment_root=str(root),
                manifest_sha256=sha(manifests[name]), schedule_sha256=sha(schedules[name]),
                external_validation_split='external_validation', selection_split='inner spatial-block dev',
                world_size=2, global_batch=32, temporal_variant=temporal,
                screening_only=args.stage=='screen', subset_grouping='block x day (conservative, not native IDs)')
            path = bundle/f'{run_id}.py'
            run.dump(str(path))
            rows.append(dict(run_id=run_id,variant=variant,seed=seed,steps=args.steps,config=str(path),
                manifest=str(manifests[name]),schedule=str(schedules[name]),
                train_windows=len(pools[name]),normalization_source=stats[name]['normalization_source_ids_sha256'],
                status='prepared',stage=args.stage))
    pd.DataFrame(rows).to_csv(bundle/'run_matrix.csv',index=False)
    initial = [r['config'] for r in rows if r['seed']==args.seeds[0] and r['variant'] in ('A0','A1','A2','A3')]
    (bundle/'initial_configs.txt').write_text('\n'.join(initial)+'\n')
    write_json(bundle/'protocol.json',dict(source_manifest=str(manifest_path),source_sha256=sha(manifest_path),
        seeds=args.seeds,stage=args.stage,steps=args.steps,global_batch=32,inner_dev_fraction_blocks=.2,
        external_validation_used_for_selection=False,test_evaluated=False,
        initial_submission='A0-A3 one seed screening after consistency and two-GPU smoke gates',
        deferred='B/C/D and multi-seed full training are prepared, not automatically submitted',
        limitations=['Native SM pixel IDs unavailable: conservative block x date grouping',
                    'B subsets randomized by conservative groups; geography/season balance reported, not guaranteed',
                    'C runs require actual common-support eligibility audit before submission']))
    print(f'Prepared {len(rows)} runs at {bundle}',flush=True)


if __name__ == '__main__':
    main()
