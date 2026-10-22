"""Disposable BF16 training benchmark; never saves trained weights or metrics.

Uses cached real validation samples and historical training moments solely for
loss scaling. Timing includes collation, H2D, forward, backward, clipping and
AdamW, but excludes Zarr I/O. It is not a training run or accuracy evaluation.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract
from hydrostate.datasets.collate import hydro_collate
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, task_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch', type=int, required=True, choices=[2, 4, 8, 16])
    parser.add_argument('--checkpoint', required=True, help='Historical moments only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = True
    init_default_scope('hydrostate')
    cfg = Config.fromfile(args.config)
    configure_data_contract(cfg)
    historical = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old = Config.fromstring(historical['meta']['cfg'], '.py')
    moments = copy.deepcopy(old.model.target_statistics)
    specs = task_config(LEGACY_TASKS)
    moments['tasks'] = {task: moments['tasks'][task] for task in specs}
    if moments['version'] == 2:
        moments['task_specs'] = specs
    cfg.model.target_statistics = moments
    del historical, old
    dataset = HydroStateZarrDataset(**{
        k: v for k, v in cfg.val_dataloader.dataset.items() if k != 'type'
    })
    pool, ids, skipped = [], [], 0
    for i in range(min(len(dataset), 320)):
        try:
            sample = dataset[i]
        except (FileNotFoundError, KeyError):
            skipped += 1
            continue
        pool.append(sample)
        ids.append(str(dataset.table.iloc[i].sample_id))
        if len(pool) == 64:
            break
    if len(pool) != 64:
        raise RuntimeError(f'Need 64 real samples, found {len(pool)}')
    print('Loaded 64 real samples; building', args.config, 'batch', args.batch, flush=True)
    model = MODELS.build(cfg.model).cuda().train()
    opt_cfg = copy.deepcopy(cfg.optim_wrapper)
    opt_cfg.accumulative_counts = 16 // args.batch
    optimizer = build_optim_wrapper(model, opt_cfg)
    optimizer.initialize_count_status(model, 0, 5 * (16 // args.batch))
    trainable = [p for p in model.parameters() if p.requires_grad]
    result = dict(config=args.config, batch=args.batch, accumulation=16 // args.batch,
                  effective_batch=16, gpu=torch.cuda.get_device_name(),
                  gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  trainable_parameters=sum(p.numel() for p in trainable),
                  sample_ids=ids, skipped_samples=skipped,
                  input_shapes={k: list(v.shape) for k, v in pool[0]['inputs'].items()
                                if torch.is_tensor(v)},
                  scope='cached real validation samples; historical train moments; '
                        'disposable optimizer updates, no saved weights; excludes Zarr I/O',
                  code_commit=os.environ.get('BENCHMARK_COMMIT'), status='running')
    times = []
    try:
        # One full effective-batch update warms up kernels and optimizer state.
        for update in range(5):
            torch.cuda.synchronize()
            if update == 1:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for offset in range(0, 16, args.batch):
                indices = [((max(update - 1, 0) * 16) + offset + j) % 64
                           for j in range(args.batch)]
                raw = hydro_collate([copy.deepcopy(pool[j]) for j in indices])
                losses = model.train_step(raw, optimizer)
                if not all(torch.isfinite(v).all() for v in losses.values()
                           if torch.is_tensor(v)):
                    raise RuntimeError('Nonfinite loss')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if update:
                times.append(elapsed)
            print('update', update, 'seconds', round(elapsed, 3), flush=True)
        if not all(torch.isfinite(p).all().item() for p in trainable):
            raise RuntimeError('Nonfinite parameters after optimizer updates')
        result.update(status='ok', update_seconds=times,
                      samples_per_second=64 / sum(times),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='oom', error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sample_ids'}), flush=True)


if __name__ == '__main__':
    main()
