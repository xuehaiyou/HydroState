"""Paired interventions on a trained CGC, reusing its encoder output per window."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from mmengine.config import Config

from hydrostate.cli import Runner, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.models.site_supervision import station_values


@torch.inference_mode()
def run(config, checkpoint, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(config)
    if cfg.val_dataloader.dataset.split != 'validation':
        raise ValueError('Development validation required')
    cfg.train_dataloader.dataset.training = False
    cfg.work_dir = str(directory / 'log')
    configure_target_statistics(cfg, training=True)
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint, strict=True)
    model = runner.model.eval()
    gates = model.router.gates
    totals = {name: torch.zeros_like(gate.bias, dtype=torch.float64) for name, gate in gates.items()}
    counts = dict.fromkeys(gates, 0)

    def accumulate(name):
        def hook(module, args, logits):
            totals[name].add_(logits.double().softmax(-1).sum(0))
            counts[name] += len(logits)
        return hook

    with ExitStack() as stack:
        for name, gate in gates.items():
            stack.enter_context(gate.register_forward_hook(accumulate(name)))
        for raw in runner.train_dataloader:
            batch = model.data_preprocessor(raw, training=False)
            model._forward(batch['inputs'])
    means = {name: total / counts[name] for name, total in totals.items()}
    atomic_json(directory / 'gate_means.json', {k: v.tolist() for k, v in means.items()})
    rows = []
    for raw in runner.val_dataloader:
        batch = model.data_preprocessor(raw, training=False)
        inputs, samples = batch['inputs'], batch['data_samples']
        levels = model.encoder(inputs)
        with patch.object(model.encoder, 'forward', return_value=levels):
            for mode in ('dynamic', 'train_mean', 'uniform', 'private_only', 'shared_only'):
                with ExitStack() as stack:
                    for name, gate in gates.items():
                        if mode == 'dynamic':
                            continue
                        def intervene(module, args, logits, name=name):
                            if mode == 'train_mean':
                                return means[name].log().to(logits).expand_as(logits)
                            if mode == 'uniform':
                                return torch.zeros_like(logits)
                            altered = logits.clone()
                            altered[:, slice(None, -1) if mode == 'private_only' else -1] = -torch.inf
                            return altered
                        stack.enter_context(gate.register_forward_hook(intervene))
                    fields = model._forward(inputs)
                    for task, (pred, target, valid) in station_values(fields, samples).items():
                        for i, j in valid.nonzero().tolist():
                            info = samples['metainfo'][i]
                            rows.append(dict(mode=mode, task=task, sample_id=info['sample_id'],
                                             site=info['spatial_group_id'], observation=j,
                                             pred=pred[i,j].item(), target=target[i,j].item()))
    pd.DataFrame(rows).to_parquet(directory / 'paired.parquet', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('config', 'checkpoint', 'directory'):
        parser.add_argument(argument)
    run(**vars(parser.parse_args()))
