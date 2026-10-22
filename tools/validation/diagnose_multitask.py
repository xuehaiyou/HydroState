"""Offline v2 diagnostics: full-split sufficient statistics and local gradients.

Never trains or evaluates the test split. Checkpoints must be frozen local copies.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader, Subset

import hydrostate  # noqa: F401
from hydrostate.datasets.collate import hydro_collate
from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.registry import DATASETS, MODELS

TASKS = ('water', 'soil_moisture', 'et')
FIELDS = ('count', 'abs', 'sq', 'err', 'y', 'y2')


def metrics(frame):
    n, absolute, sq, err, y, y2 = frame[list(FIELDS)].sum().to_numpy(float)
    if n == 0:
        return {'count': 0}
    variance = y2 - y * y / n
    return dict(count=int(n), mae=absolute/n, rmse=math.sqrt(sq/n), bias=err/n,
                r2=1-sq/variance if variance > 1e-12 else None,
                unbiased_rmse=math.sqrt(max(0, sq/n-(err/n)**2)))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def loader(dataset, indices=None, batch=16, workers=4):
    return DataLoader(dataset if indices is None else Subset(dataset, indices),
                      batch_size=batch, shuffle=False, num_workers=workers,
                      collate_fn=hydro_collate, pin_memory=True,
                      multiprocessing_context='spawn' if workers else None,
                      timeout=180 if workers else 0)


def evaluate(model, dataset, split, out, batch, workers):
    path = out / f'{split}_windows.csv'
    records = []
    offset = 0
    with torch.inference_mode():
        for step, raw in enumerate(loader(dataset, batch=batch, workers=workers)):
            data = model.data_preprocessor(raw, training=False)
            outputs = model._forward(data['inputs'])
            b = next(iter(outputs.values())).shape[0]
            available = data['inputs']['sensor_valid'].any(dim=1).cpu().numpy()
            for task in TASKS:
                pred, target, valid = supervision_values(task, outputs[task], data['data_samples'])
                pred, target, valid = torch.broadcast_tensors(pred, target, valid)
                for j in range(b):
                    p, y = pred[j][valid[j]].double(), target[j][valid[j]].double()
                    e = p-y
                    row = dataset.table.iloc[offset+j]
                    records.append(dict(index=offset+j, sample_id=str(row.sample_id), task=task,
                        count=y.numel(), abs=e.abs().sum().item(), sq=e.square().sum().item(),
                        err=e.sum().item(), y=y.sum().item(), y2=y.square().sum().item(),
                        lon=float(row.lon), lat=float(row.lat), climate=str(row.climate_class),
                        month=int(pd.Timestamp(row.target_time).month),
                        block=f'{math.floor(float(row.lon)/5)}_{math.floor(float(row.lat)/5)}',
                        modalities=''.join(str(int(x)) for x in available[j])))
            offset += b
            if step % 10 == 0:
                print(f'EVAL {out.name} {split} {offset}/{len(dataset)}', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(path, index=False)
    result = {}
    groups = []
    for task in TASKS:
        f = frame[(frame.task == task) & (frame['count'] > 0)]
        result[task] = metrics(f)
        result[task]['valid_windows'] = len(f)
        result[task]['window_macro_rmse'] = float(np.sqrt((f.sq/f['count']).mean())) if len(f) else None
        for category in ('climate', 'month', 'modalities', 'block'):
            for name, g in f.groupby(category):
                groups.append(dict(task=task, category=category, group=str(name), windows=len(g), **metrics(g)))
        block_stats = f.groupby('block')[list(FIELDS)].sum()
        rng = np.random.default_rng(3407)
        boots = []
        if len(block_stats) > 1:
            for _ in range(1000):
                sampled = block_stats.iloc[rng.integers(len(block_stats), size=len(block_stats))]
                boots.append(metrics(sampled)['rmse'])
        result[task]['spatial_block_rmse_ci95'] = np.quantile(boots, [.025, .975]).tolist() if boots else None
        result[task]['spatial_blocks'] = len(block_stats)
    pd.DataFrame(groups).to_csv(out / f'{split}_groups.csv', index=False)
    dump(out / f'{split}_metrics.json', result)
    return frame, result


def consistency(model, dataset, frame, out, workers, limit=64):
    # Stratify by actual modality pattern and SM validity, then keep <=64 windows.
    pivot = frame.pivot(index='index', columns='task', values='count')
    info = frame.drop_duplicates('index').set_index('index')[['modalities']].copy()
    info['sm'] = pivot.soil_moisture > 0
    strata = []
    for _, group in info.groupby(['modalities', 'sm']):
        strata.append(group.sample(min(8, len(group)), random_state=3407).index.tolist())
    chosen = [group[j] for j in range(8) for group in strata if j < len(group)]
    rng = np.random.default_rng(3407)
    rest = [i for i in info.index if i not in chosen]
    chosen = chosen[:limit] + rng.permutation(rest).tolist()[:max(0, limit-len(chosen))]
    variants = [('single', 1, chosen), ('repeat', 1, chosen), ('batch2', 2, chosen),
                ('batch16', 16, chosen), ('batch32', 32, chosen),
                ('shuffle16', 16, rng.permutation(chosen).tolist())]
    reference = {}
    records = []
    with torch.inference_mode():
        for name, batch, indices in variants:
            offset = 0
            for raw in loader(dataset, indices, batch, workers):
                data = model.data_preprocessor(raw, training=False)
                outputs = model._forward(data['inputs'])
                for task in TASKS:
                    pred, _, valid = supervision_values(task, outputs[task], data['data_samples'])
                    pred, valid = torch.broadcast_tensors(pred, valid)
                    for j in range(len(pred)):
                        index = indices[offset+j]
                        value = pred[j][valid[j]].float().cpu()
                        if name == 'single':
                            reference[index, task] = value
                        elif len(value):
                            error = value-reference[index, task]
                            records.append(dict(variant=name, index=index, task=task,
                                max_abs=error.abs().max().item(), rmse=error.square().mean().sqrt().item()))
                offset += len(next(iter(outputs.values())))
    pd.DataFrame(records).to_csv(out / 'batch_consistency.csv', index=False)
    dump(out / 'consistency_indices.json', chosen)
    print('CONSISTENCY complete', flush=True)


def gradient_probe(model, dataset, out, batches, workers):
    # Exact parameter gradients of two named shared modules, not whole-network gradients.
    modules = {'encoder_last_block': model.encoder.backbone.model.blocks[-1],
               'shared_decoder': model.decoder}
    for p in model.parameters():
        p.requires_grad_(False)
    groups = {}
    for name, module in modules.items():
        groups[name] = list(module.parameters())
        for p in groups[name]:
            p.requires_grad_(True)
    params = sum(groups.values(), [])
    dump(out / 'gradient_scope.json', {name: sum(p.numel() for p in ps) for name, ps in groups.items()})
    records = []
    std = model.loss_module.target_std
    # Replay global replacement sampling, combining both DDP streams into 32 windows.
    for seed in (3407, 3408, 3409):
        gen = torch.Generator().manual_seed(seed)
        indices = torch.multinomial(dataset.sampling_weights(), batches*32, replacement=True, generator=gen).tolist()
        for step, raw in enumerate(loader(dataset, indices, 32, workers)):
            data = model.data_preprocessor(raw, training=False)
            outputs = model._forward(data['inputs'])
            grads = {name: [] for name in groups}
            counts, windows, scales = [], [], []
            for ti, task in enumerate(TASKS):
                pred, target, valid = supervision_values(task, outputs[task], data['data_samples'])
                pred, target, valid = torch.broadcast_tensors(pred, target, valid)
                count = int(valid.sum())
                counts.append(count)
                windows.append(int(valid.reshape(len(valid), -1).any(1).sum()))
                # NLL shared gradient = standardized-MSE gradient / (2*variance).
                scales.append(float(0.5/model.loss_module.variance(task)))
                objective = ((pred[valid]-target[valid])/std[ti]).square().sum()/max(count, 1)
                gs = torch.autograd.grad(objective, params, retain_graph=ti < 2, allow_unused=True)
                offset = 0
                for name, ps in groups.items():
                    vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                        for p, g in zip(ps, gs[offset:offset+len(ps)])])
                    grads[name].append(vector)
                    offset += len(ps)
            for name, vectors in grads.items():
                gram = torch.stack([torch.stack([torch.dot(a, b) for b in vectors]) for a in vectors]).double().cpu().numpy()
                for kind, weights in [('mse', np.ones(3)), ('nll', np.array(scales))]:
                    matrix = gram * weights[:, None] * weights[None, :]
                    norms = np.sqrt(np.maximum(0, matrix.diagonal()))
                    for i, task in enumerate(TASKS):
                        row = dict(seed=seed, batch=step, scope=name, objective=kind, task=task,
                                   valid_count=counts[i], valid_windows=windows[i], norm=float(norms[i]),
                                   dot_combined=float(matrix[i].sum()))
                        for j, other in enumerate(TASKS):
                            row[f'cos_{other}'] = float(matrix[i, j]/(norms[i]*norms[j])) if norms[i]*norms[j] > 0 else None
                        records.append(row)
            del outputs, grads, gs, vectors, vector, objective
            if step % 10 == 0:
                print(f'GRAD {out.name} seed={seed} batch={step}/{batches}', flush=True)
        pd.DataFrame(records).to_csv(out / 'gradients.csv', index=False)
    for p in model.parameters():
        p.requires_grad_(True)


def sampling_audit(dataset, frame, out):
    counts = frame.pivot(index='index', columns='task', values='count').reindex(range(len(dataset)))
    valid = counts[TASKS[0]].notna()
    assert valid.all(), 'Audit must cover every training window'
    present = counts.gt(0)
    patterns = present.apply(lambda r: ''.join(t[0] for t in TASKS if r[t]) or 'none', axis=1).value_counts().to_dict()
    records = []
    for epoch in range(5):
        indices = torch.multinomial(dataset.sampling_weights(), len(dataset), replacement=True,
                                    generator=torch.Generator().manual_seed(3407+epoch)).numpy()
        for start in range(0, len(indices), 32):
            chosen = indices[start:start+32]
            records.append(dict(epoch=epoch, batch=start//32, windows=len(chosen),
                                **{t: int(present.iloc[chosen][t].sum()) for t in TASKS}))
    pd.DataFrame(records).to_csv(out / 'sampler_batches.csv', index=False)
    dump(out / 'validity_patterns.json', patterns)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gradient-batches-per-seed', type=int, default=34)
    p.add_argument('--max-windows', type=int, default=0, help='Preflight only: seeded random subset per split')
    p.add_argument('--consistency-windows', type=int, default=64)
    args = p.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.config)
    init_default_scope('hydrostate')
    datasets = {}
    for split in ('train', 'validation'):
        dc = dict(cfg.train_dataloader.dataset)
        dc.update(split=split, training=False, flip_probability=0)
        datasets[split] = DATASETS.build(dc)
        if args.max_windows:
            datasets[split].table = datasets[split].table.sample(
                min(args.max_windows, len(datasets[split])), random_state=3407).reset_index(drop=True)
    dump(args.out / 'conditions.json', dict(max_windows=args.max_windows,
        split_windows={s: len(d) for s, d in datasets.items()}, batch_size=args.batch_size,
        gradient_batches_per_seed=args.gradient_batches_per_seed, consistency_windows=args.consistency_windows,
        precision='FP32', mode='eval', seed=3407))
    model = MODELS.build(cfg.model).cuda().eval()
    results = {}
    for checkpoint in args.checkpoints:
        name = Path(checkpoint).stem
        out = args.out / name
        out.mkdir(exist_ok=True)
        load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
        model.eval()
        results[name] = {}
        frames = {}
        for split, dataset in datasets.items():
            frames[split], results[name][split] = evaluate(model, dataset, split, out, args.batch_size, args.workers)
        dump(args.out / 'metrics.json', results)
        sampling_audit(datasets['train'], frames['train'], out)
        consistency(model, datasets['validation'], frames['validation'], out, args.workers, args.consistency_windows)
        gradient_probe(model, datasets['train'], out, args.gradient_batches_per_seed, args.workers)
    dump(args.out / 'completed.json', {'checkpoints': args.checkpoints, 'test_evaluated': False,
        'gradient_caveat': 'FP32 eval-mode single-GPU global batches; scopes only, not AdamW updates or exact DDP forwards',
        'block_caveat': '5-degree coordinate blocks used for bootstrap; not independent native observation IDs'})
    print('DIAGNOSTICS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
