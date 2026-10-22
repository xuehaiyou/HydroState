"""Offline sample -> pretrained model -> gradient -> checkpoint -> inference check.

No GEE calls or persistent model weights. Unit statistics, when explicitly
requested, test connectivity only and must never be used for accuracy claims.
"""

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.input_products import L2_PROFILE
from hydrostate.datasets.collate import hydro_collate
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.inference.predict import TASKS, WeightedMosaic, load_model, masked_outputs
from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
from hydrostate.registry import MODELS
from hydrostate.tasks import resolve_tasks, task_config


def validate(args):
    import hydrostate.models  # noqa: F401

    init_default_scope('hydrostate')
    torch.set_num_threads(args.threads)
    cfg = Config.fromfile(args.config)
    configure_data_contract(cfg)
    if cfg.model.encoder.get('input_profile') != L2_PROFILE:
        raise ValueError('Use a configuration with the explicit L2 input profile')
    if args.unit_statistics:
        specs = resolve_tasks(cfg.model.get('tasks'))
        cfg.model.target_statistics = dict(
            version=2, split='train', ddof=0, support_version=CONFIGURED_SUPPORT_VERSION,
            training_fingerprint='0' * 64, task_specs=task_config(specs),
            data_contract=dict(input_profile=L2_PROFILE), validation_only=True,
            tasks={k: dict(count=2, mean=0, std=1) for k in specs})
    else:
        configure_target_statistics(cfg, training=True)
    cfg.model.encoder.model_path = str(args.pretrained.resolve())
    if not (args.pretrained / 'weights.pth').is_file():
        raise FileNotFoundError('A local OlmoEarth weights.pth is required; no automatic downloads')
    config = dict(cfg.train_dataloader.dataset)
    config.pop('type')
    config.update(training=False, flip_probability=0)
    dataset = HydroStateZarrDataset(**config)
    model = MODELS.build(cfg.model).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    report = dict(input_profile=L2_PROFILE, device=args.device, model=cfg.model.type,
                  statistics='validation-only unit scales' if args.unit_statistics else 'training',
                  tasks=list(model.tasks), records=[], checkpoint_reloaded=False)
    for index in range(min(args.limit, len(dataset))):
        started = time.monotonic()
        sample = dataset[index]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data = model.data_preprocessor(hydro_collate([copy.deepcopy(sample)]), training=True)
        weight = model.encoder.landsat_l2_adapter.projection.weight
        before = weight.detach().clone()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            losses = model(**data, mode='loss')
            objective = sum(value for key, value in losses.items() if key.startswith('loss_'))
        objective.backward()
        if not torch.isfinite(objective) or any(not torch.isfinite(p.grad).all()
                                                for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite loss or gradient')
        gradient = 0 if weight.grad is None else float(weight.grad.abs().sum())
        if sample['inputs']['landsat_l2_valid'].any() and gradient == 0:
            raise ValueError('L2 observations did not reach the learning objective')
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        record = dict(sample_id=sample['data_samples']['metainfo']['sample_id'],
                      loss=float(objective.detach()), l2_gradient_l1=gradient,
                      l2_weight_changed=not torch.equal(before, weight),
                      elapsed_seconds=time.monotonic() - started)
        report['records'].append(record)
        print(json.dumps(record), flush=True)
    model.eval()
    inputs = hydro_collate([copy.deepcopy(sample)])['inputs']
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        if tuple(model.tasks) == TASKS:
            expected = masked_outputs(model, inputs)
        else:
            prepared = model.data_preprocessor(hydro_collate([copy.deepcopy(sample)]))
            expected = {k: v.float().cpu().numpy() for k, v in model(prepared['inputs']).items()}
    with tempfile.TemporaryDirectory(prefix='hydrostate-l2-check-') as folder:
        folder = Path(folder)
        checkpoint, stats, model_config = [folder / p for p in (
            'model.pth', 'stats.json', 'model.py')]
        torch.save(dict(state_dict=model.state_dict()), checkpoint)
        stats.write_text(json.dumps(dict(cfg.model.target_statistics)))
        Config(dict(model=cfg.model, input_profile=L2_PROFILE)).dump(str(model_config))
        del model, optimizer, data, losses, objective, weight
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        if tuple(report['tasks']) == TASKS:
            plan = dict(config=dict(input_profile=L2_PROFILE, pretrained=str(args.pretrained),
                                    target_statistics=str(stats)))
            restored, _ = load_model(plan, checkpoint=checkpoint, model_config=model_config,
                                     device=args.device)
            with torch.inference_mode(), torch.autocast(
                    'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                actual = masked_outputs(restored, inputs)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-5, rtol=1e-5)
            mosaic = WeightedMosaic((32, 32), 32)
            mosaic.add(0, 0, {k: v[0, 4:36, 4:36] for k, v in actual.items()})
            report['mosaic_finite_pixels'] = {k: int(np.isfinite(v).sum())
                                              for k, v in mosaic.finish().items()}
            if args.region_inputs:
                with np.load(args.region_inputs, allow_pickle=False) as stored:
                    regional = {k: torch.from_numpy(stored[k].copy()).unsqueeze(0)
                                for k in stored if not k.startswith('query_')}
                    regional['target_time'] = {
                        k: torch.from_numpy(stored['query_' + k].copy()).unsqueeze(0)
                        for k in TASKS}
                with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                    regional_output = masked_outputs(restored, regional)
                region_mosaic = WeightedMosaic((32, 32), 32)
                region_mosaic.add(0, 0, {k: v[0, 4:36, 4:36]
                                         for k, v in regional_output.items()})
                report['regional_window'] = dict(
                    source=str(args.region_inputs),
                    finite_pixels={k: int(np.isfinite(v).sum())
                                   for k, v in region_mosaic.finish().items()})
        else:
            restored = MODELS.build(cfg.model).to(args.device).eval()
            restored.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                                weights_only=False)['state_dict'], strict=True)
            with torch.inference_mode(), torch.autocast(
                    'cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                prepared = restored.data_preprocessor(hydro_collate([copy.deepcopy(sample)]))
                actual = restored(prepared['inputs'])
            for key in expected:
                np.testing.assert_allclose(actual[key].float().cpu().numpy(), expected[key],
                                           atol=1e-5, rtol=1e-5)
        report['checkpoint_reloaded'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/legacy/hydrostate_multilevel_temporal_l2.py')
    parser.add_argument('--pretrained', type=Path, default=Path('pretrained/OlmoEarth-v1_2-Base'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--region-inputs', type=Path,
                        help='Optional NPZ of one assembled local window plus query_TASK bounds')
    parser.add_argument('--unit-statistics', action='store_true',
                        help='Connectivity check only while full training statistics are pending')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.threads < 1:
        parser.error('limit and threads must be positive')
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
