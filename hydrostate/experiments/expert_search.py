"""Station-selected architecture screening: bounded probes, then measured submissions."""

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json

from .joint_debug import execute, safe
from .search_launch import estimate_budget
from .search_recipe import configure_search

VARIANTS = {
    'A': 'hydrostate_joint_shared',
    'B': 'hydrostate_joint_independent',
    'CGC': 'hydrostate_joint_cgc',
    'CGC_matched': 'hydrostate_joint_cgc_matched',
    'PLE': 'hydrostate_joint_ple',
    'MLoRE_complete': 'hydrostate_joint_mlore_complete',
    'AdapterFusion': 'hydrostate_joint_adapterfusion_scratch',
    'CoSA_Router': 'hydrostate_joint',
}
EFFECTIVE_BATCH = 10


def prepare(root, selection_root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('x') as stream:
        stream.write('One frozen screening plan per directory\n')
    for variant, name in VARIANTS.items():
        directory = root / variant
        (directory / 'debug/logs').mkdir(parents=True)
        cfg = configure_search(Config.fromfile(f'configs/{name}.py'), selection_root,
                               work_dir=directory / 'work', batch_size=5, accumulation=2,
                               station_selection=True)
        # The smoke's test API checks reload on DEVELOPMENT data, never sealed test data.
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        cfg.dump(str(directory / 'base.py'))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    atomic_json(root / 'plan.json', dict(variants=VARIANTS, code_commit=commit,
        selection_root=str(Path(selection_root).resolve()), epochs=6,
        effective_batch=EFFECTIVE_BATCH, selection='site_score',
        scope='Fixed development screening; no sealed test evaluation'))


def accept(root, variant):
    directory = Path(root).resolve() / variant
    debug = directory / 'debug'
    with (debug / 'accept.lock').open('x') as stream:
        stream.write('One GPU acceptance attempt\n')
    candidates = []
    # Divisors of the fixed effective batch preserve the optimizer-update budget.
    for batch in (2, 5, 10):
        report = debug / f'benchmark_{batch}.json'
        result = execute(debug, ['tools/benchmarks/joint_training.py', '--config',
            str(directory / 'base.py'), '--output', str(report), '--batch-size', str(batch),
            '--steps', '6', '--workers', '2'], f'benchmark_{batch}', report)
        if not safe(result):
            break
        candidates.append(batch)
    if not candidates:
        report = debug / 'benchmark_1.json'
        result = execute(debug, ['tools/benchmarks/joint_training.py', '--config',
            str(directory / 'base.py'), '--output', str(report), '--batch-size', '1',
            '--steps', '6', '--workers', '2'], 'benchmark_1', report)
        if safe(result):
            candidates.append(1)
    for batch in reversed(candidates):
        accumulation = EFFECTIVE_BATCH // batch
        cfg = Config.fromfile(str(directory / 'base.py'))
        cfg.train_dataloader.batch_size = cfg.val_dataloader.batch_size = batch
        cfg.test_dataloader.batch_size = batch
        cfg.optim_wrapper.accumulative_counts = accumulation
        cfg.param_scheduler[0].end = 100 * accumulation
        config = directory / f'candidate_{batch}.py'
        cfg.dump(str(config))
        output = debug / f'smoke_{batch}'
        result = execute(debug, ['tools/experiments/verify_joint_gpu.py', '--config',
            str(config), '--output', str(output), '--batch-size', str(batch),
            '--accumulation', str(accumulation), '--train-steps', str(max(4, accumulation)),
            '--workers', '2'], f'smoke_{batch}', output / 'result.json',
            failure=output / 'failure.json')
        if not safe(result):
            continue
        cfg.dump(str(directory / 'config.py'))
        atomic_json(debug / 'passed.json', dict(status='passed', variant=variant,
            batch_size=batch, accumulation=accumulation, effective_batch=EFFECTIVE_BATCH,
            benchmark=str(debug / f'benchmark_{batch}.json'), smoke=str(output / 'result.json')))
        return
    raise RuntimeError(f'No safe real-data acceptance for {variant}')


def submit_once(record, command, metadata):
    if record.exists():
        prior = json.loads(record.read_text())
        if prior['status'] != 'accepted':
            raise RuntimeError(f'Inspect prior submission before retrying: {record}')
        return prior['job_id']
    item = dict(metadata, status='submitting', command=command)
    atomic_json(record, item)
    result = subprocess.run(command, capture_output=True, text=True)
    item.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    if result.returncode == 0:
        item.update(status='accepted', job_id=result.stdout.strip().split(';')[0])
    else:
        item['status'] = 'rejected'
    atomic_json(record, item)
    result.check_returncode()
    return item['job_id']


def environment_script(release, env, command):
    lines = ['#!/bin/bash', 'set -euo pipefail', f'cd {shlex.quote(str(release))}']
    for key, value in env.items():
        lines.append(f'export {key}={shlex.quote(str(value))}')
    return '\n'.join([*lines, 'exec ' + shlex.join(command)]) + '\n'


def submit_training(root, partitions='l40s'):
    if not partitions or set(partitions.split(',')) - {'l40s', 'c_foss_gpu'}:
        raise ValueError('Unsupported GPU partitions')
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    env = json.loads((root / 'environment.json').read_text())
    env['HYDROSTATE_SLURM_PARTITION'] = partitions
    release = Path(env['HYDROSTATE_CODE_ROOT'])
    if plan['code_commit'] != env['HYDROSTATE_CODE_COMMIT']:
        raise ValueError('Training and acceptance must use the same frozen commit')
    calibration = json.loads((root / 'timing_calibration.json').read_text())
    previous = None
    for variant in plan['variants']:
        directory = root / variant
        receipt = json.loads((directory / 'debug/submission.json').read_text())
        job = receipt['job_id']
        state = subprocess.check_output(['sacct', '-n', '-P', '-j', job,
            '--format=JobIDRaw,State,ExitCode'], text=True)
        if [job, 'COMPLETED', '0:0'] not in [r.split('|')[:3] for r in state.splitlines()]:
            raise ValueError(f'Acceptance incomplete for {variant}: {state}')
        passed = json.loads((directory / 'debug/passed.json').read_text())
        bench = json.loads(Path(passed['benchmark']).read_text())
        smoke = json.loads(Path(passed['smoke']).read_text())
        adjusted = dict(bench)
        # Calibrate short-probe extrapolation with the previous completed screening,
        # then apply the standard 10% walltime margin in estimate_budget.
        adjusted['train_seconds_per_sample'] *= calibration['train_factor']
        adjusted['validation_seconds_per_sample'] *= calibration['validation_factor']
        budget = estimate_budget(adjusted, smoke)
        config = Config.fromfile(str(directory / 'config.py'))
        if (config.default_hooks.checkpoint.save_best != 'site_score'
                or config.train_dataloader.batch_size * config.optim_wrapper.accumulative_counts
                != EFFECTIVE_BATCH):
            raise ValueError('Unmatched station selection or effective batch')
        script = directory / 'train.sh'
        script.write_text(environment_script(release, env,
            [sys.executable, 'tools/train.py', str(directory / 'config.py')]))
        cmd = ['sbatch', '--parsable', f'--job-name=hs_station_{variant}',
            f'--partition={partitions}', '--qos=gpu', '--account=geog_geors', '--nodes=1',
            '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=4', '--mem=64G',
            f'--time={budget["walltime"]}', f'--output={directory}/%j.out',
            f'--error={directory}/%j.err']
        # A failed independent model must not prevent the other candidates running.
        if previous:
            cmd.append(f'--dependency=afterany:{previous}')
        cmd.append(str(script))
        previous = submit_once(directory / 'submission.json', cmd,
            dict(code_commit=plan['code_commit'], partitions=partitions,
                 debug_job=job, budget=budget,
                 batch_size=passed['batch_size'], accumulation=passed['accumulation'],
                 epochs=6, selection='site_score'))
        print(variant, previous, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'accept', 'submit-training'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--selection-root')
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--partitions', default='l40s')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not args.selection_root:
            parser.error('--selection-root is required')
        prepare(args.root, args.selection_root)
    elif args.stage == 'accept':
        if not args.variant:
            parser.error('--variant is required')
        accept(args.root, args.variant)
    else:
        submit_training(args.root, args.partitions)
