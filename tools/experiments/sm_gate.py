#!/usr/bin/env python
"""GPU prerequisite: token-layout grouping must match singleton predictions."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader, Subset
import hydrostate.experiments.sm_causal  # noqa
from hydrostate.registry import MODELS, DATASETS
from hydrostate.datasets.zarr_dataset import hydro_collate
from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.normalization import TASKS


def code_hash():
    return hashlib.sha256(Path('hydrostate/experiments/sm_causal.py').read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--windows', type=int, default=64)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    passed = args.out / 'passed.json'
    if passed.exists():
        raise FileExistsError(passed)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    init_default_scope('hydrostate')
    cfg = Config.fromfile(args.snapshot / 'hydrostate_l40s.py')
    dc = dict(cfg.val_dataloader.dataset)
    dc.update(training=False, flip_probability=0)
    dataset = DATASETS.build(dc)
    frame = pd.read_csv(args.snapshot / 'job_4172645/latest/validation_windows.csv')
    sm = frame[(frame.task == 'soil_moisture') & (frame['count'] > 0)]
    candidates = []
    for _, group in sm.groupby('modalities'):
        candidates.extend(group.sample(min(8, len(group)), random_state=3407)['index'].tolist())
    for _, group in frame.drop_duplicates('index').groupby('modalities'):
        candidates.extend(group.sample(min(8, len(group)), random_state=3407)['index'].tolist())
    candidates.extend(np.random.default_rng(3407).permutation(len(dataset)).tolist())
    chosen = list(dict.fromkeys(candidates))[:args.windows]
    records = []
    checkpoint_hashes = {}
    for checkpoint in ('best.pth', 'latest.pth'):
        path = args.snapshot / checkpoint
        checkpoint_hashes[checkpoint] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        model_cfg = dict(cfg.model)
        model_cfg['encoder'] = dict(model_cfg['encoder'], type='CausalOlmoEarthEncoder')
        model = MODELS.build(model_cfg).cuda().eval()
        load_checkpoint(model, str(path), map_location='cpu', strict=True)
        reference = {}
        variants = [('single', 1, chosen), ('repeat', 1, chosen), ('batch2', 2, chosen),
                    ('batch16', 16, chosen), ('batch32', 32, chosen),
                    ('shuffle16', 16, chosen[::-1])]
        for variant, batch_size, indices in variants:
            offset = 0
            with torch.inference_mode():
                loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                                    collate_fn=hydro_collate, num_workers=0)
                for raw in loader:
                    data = model.data_preprocessor(raw, False)
                    outputs = model._forward(data['inputs'])
                    for task in TASKS:
                        pred, _, valid = supervision_values(task, outputs[task], data['data_samples'])
                        pred, valid = torch.broadcast_tensors(pred, valid)
                        for j in range(len(pred)):
                            key = (indices[offset+j], task)
                            values = pred[j][valid[j]].float().cpu()
                            if variant == 'single':
                                reference[key] = values
                            elif values.numel():
                                ref = reference[key]
                                ok = bool(torch.allclose(values, ref, atol=1e-5, rtol=1e-4))
                                records.append(dict(checkpoint=checkpoint, variant=variant,
                                    index=key[0], task=task, max_abs=float((values-ref).abs().max()), passed=ok))
                    offset += len(next(iter(outputs.values())))
            print(checkpoint, variant, 'done', flush=True)
            pd.DataFrame(records).to_csv(args.out / 'consistency.csv', index=False)
        # Backpropagation check on SM-bearing windows; training dropout is not
        # expected to give exact cross-batch equality without matched RNG streams.
        model.train()
        raw = hydro_collate([dataset[int(i)] for i in sm['index'].iloc[:2]])
        data = model.data_preprocessor(raw, True)
        losses = model.loss(data['inputs'], data['data_samples'])
        loss = sum(v for k, v in losses.items() if k.startswith('loss_'))
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        records.append(dict(checkpoint=checkpoint, variant='backward', index=-1,
                            task='all', max_abs=0, passed=finite))
        del model, data, outputs, loss, losses
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(args.out / 'consistency.csv', index=False)
    report = dict(passed=all(r['passed'] for r in records), component_sha256=code_hash(),
                  checkpoint_sha256=checkpoint_hashes, indices=chosen, atol=1e-5, rtol=1e-4,
                  mode='FP32 eval comparisons; separate train-mode backward smoke',
                  method='group identical modality/aligned-time mask layouts')
    (args.out / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError('Consistency or backward gate failed; training must not run')
    passed.write_text(json.dumps(report, indent=2)+'\n')
    print('GATE PASSED', flush=True)


if __name__ == '__main__':
    main()
