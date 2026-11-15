"""Real-data acceptance for the frozen shared-LoRA fusion experiment."""

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader

from hydrostate.cli import configure_pretrained_encoder, configure_target_statistics
from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.site.training_smoke import representative_indices
from hydrostate.datasets.collate import hydro_collate
from hydrostate.registry import DATASETS, MODELS


def validate(config, output):
    if Path(output).exists():
        raise FileExistsError(output)
    cfg = Config.fromfile(config)
    init_default_scope('hydrostate')
    configure_pretrained_encoder(cfg)
    configure_target_statistics(cfg, training=True)
    torch.set_num_threads(1)
    dataset = DATASETS.build(dict(cfg.train_dataloader.dataset))
    indices = representative_indices(dataset)
    loader = DataLoader(dataset, batch_size=4, sampler=indices, num_workers=2,
                        collate_fn=hydro_collate)
    model = MODELS.build(cfg.model).cuda().train()
    batches, differences = [], []
    for raw in loader:
        batches.append(raw)
        data = model.data_preprocessor(raw, training=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model.fusion_enabled = False
            baseline = model._forward(data['inputs'])
            model.fusion_enabled = True
            actual = model._forward(data['inputs'])
        for task in baseline:
            torch.testing.assert_close(actual[task], baseline[task], atol=1e-6, rtol=1e-5)
            differences.append(float((actual[task] - baseline[task]).abs().max()))
        if len(batches) >= 3:
            break
    assert batches
    cfg.optim_wrapper.accumulative_counts = 1
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    optimizer.initialize_count_status(model, 0, len(batches) * 2)
    for raw in batches * 2:
        losses = model.train_step(raw, optimizer)
        assert all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith('fusions.') for n in trainable)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
    assert not model.encoder.training and not model.decoders.training
    assert any(f.gamma.item() != 0 for f in model.fusions.values())
    atomic_json(Path(output), dict(status='passed', samples=len(indices),
                baseline_max_absolute_difference=max(differences),
                trainable_parameters=sum(p.numel() for p in trainable.values()),
                total_parameters=sum(p.numel() for p in model.parameters()),
                gamma={t: float(f.gamma) for t, f in model.fusions.items()},
                frozen_eval=True, scope='Disposable real-data acceptance, not accuracy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hydrostate_joint_adapterfusion.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    validate(args.config, args.output)
