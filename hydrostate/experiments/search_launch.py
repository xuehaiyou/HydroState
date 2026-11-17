"""Submit sequential six-epoch screening only after all real-data probes pass."""

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json

from .gpu_memory import has_memory_headroom
from .search_recipe import configure_search


def estimate_budget(bench, smoke, epochs=6, val_interval=2):
    if (bench['status'] != 'passed' or smoke['status'] != 'passed'
            or bench['batch_size'] != smoke['micro_batch']
            or not all(has_memory_headroom(r) for r in (bench, smoke))
            or smoke['checkpoint_parameters_verified'] <= 0):
        raise ValueError('Incomplete GPU acceptance evidence')
    train = bench['train_seconds_per_sample']
    val = bench['validation_seconds_per_sample']
    if any(not math.isfinite(x) or x <= 0 for x in (train, val)):
        raise ValueError('Invalid measured throughput')
    selection = smoke['selection']
    simulated = (selection['train']['samples'] * train
                 + sum(selection[k]['samples'] for k in ('val', 'test')) * val)
    # Retain all unexplained smoke time once per epoch: initialization, metric
    # bookkeeping, writing and restoring checkpoints. No sealed test is scheduled.
    overhead = max(0., smoke['elapsed_seconds'] - simulated)
    seconds = (epochs * (bench['train_samples'] * train + overhead)
               + (epochs // val_interval) * bench['validation_samples'] * val)
    minutes = math.ceil(seconds * 1.1 / 60)
    return dict(estimated_seconds=seconds, overhead_per_epoch_seconds=overhead,
                seconds=minutes * 60, walltime=f'{minutes // 60:02d}:{minutes % 60:02d}:00')


def launch(root, debug, *, submit=False):
    root, debug = Path(root).resolve(), Path(debug).resolve()
    passed = json.loads((debug / 'passed.json').read_text())
    receipt = json.loads((debug / 'receipt.json').read_text())
    job = receipt['stdout'].strip().split(';')[0]
    state = subprocess.check_output([
        'sacct', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode'], text=True)
    if [job, 'COMPLETED', '0:0'] not in [row.split('|')[:3] for row in state.splitlines()]:
        raise ValueError('GPU acceptance job is not completed successfully')
    variants = tuple(passed['variants'])
    if variants != ('A', 'B', 'CGC', 'MLoRE'):
        raise ValueError('Require acceptance for every planned screening candidate')
    batch = passed['batch_size']
    env = json.loads((root / 'screening_environment.json').read_text())
    release = Path(env['HYDROSTATE_CODE_ROOT']).parent / receipt['commit']
    if not release.is_dir():
        raise FileNotFoundError(release)
    env.update(HYDROSTATE_CODE_ROOT=str(release), HYDROSTATE_CODE_COMMIT=receipt['commit'],
               PYTHONPATH=str(release))
    output = root / 'screen_10'
    output.mkdir(exist_ok=True)
    previous = None
    plan = {}
    for variant in variants:
        bench = json.loads((debug / f'{variant}_{batch}.json').read_text())
        smoke = json.loads((debug / f'smoke_{variant}/result.json').read_text())
        if bench['gpu'] != 'NVIDIA L40S' or bench['batch_size'] != batch:
            raise ValueError('This submission recipe requires matched L40S measurements')
        budget = estimate_budget(bench, smoke)
        directory = output / variant
        directory.mkdir(exist_ok=True)
        record = directory / 'submission.json'
        if record.exists():
            prior = json.loads(record.read_text())
            if prior['status'] != 'accepted':
                raise RuntimeError(f'Inspect interrupted submission before retrying: {record}')
            previous = prior['job_id']
            plan[variant] = prior
            continue
        cfg = configure_search(Config.fromfile(str(root / f'{variant}_10.py')), root,
                               work_dir=directory / 'work', batch_size=batch)
        cfg.dump(str(directory / 'config.py'))
        lines = ['#!/bin/bash', 'set -euo pipefail', f'cd {shlex.quote(str(release))}']
        for key, value in env.items():
            if key not in ('HYDROSTATE_WORK_DIR', 'HYDROSTATE_CONFIG', 'HYDROSTATE_SOURCE_HASHES'):
                lines.append(f'export {key}={shlex.quote(value)}')
        lines += [f'sha256sum --check --status {shlex.quote(str(debug / "source.sha256"))}',
                  'exec /home/xiaoz/miniconda/envs/hydrostate/bin/python tools/train.py '
                  + shlex.quote(str(directory / 'config.py'))]
        (directory / 'run.sh').write_text('\n'.join(lines) + '\n')
        cmd = ['sbatch', '--parsable', f'--job-name=hs_search_{variant}_10',
               '--partition=l40s', '--qos=gpu', '--account=geog_geors', '--nodes=1',
               '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=4', '--mem=48G',
               f'--time={budget["walltime"]}', f'--output={directory}/%j.out',
               f'--error={directory}/%j.err']
        if previous:
            cmd.append(f'--dependency=afterok:{previous}')
        cmd.append(str(directory / 'run.sh'))
        item = dict(status='planned', command=cmd, budget=budget, batch_size=batch,
                    epochs=6, code_commit=receipt['commit'], debug_job=job)
        if submit:
            item['status'] = 'submitting'
            atomic_json(record, item)
            result = subprocess.run(cmd, capture_output=True, text=True)
            item.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            if result.returncode == 0:
                previous = result.stdout.strip().split(';')[0]
                item.update(status='accepted', job_id=previous)
            else:
                item['status'] = 'rejected'
            atomic_json(record, item)
            result.check_returncode()
        plan[variant] = item
    atomic_json(output / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--debug', required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(launch(args.root, args.debug, submit=args.submit), indent=2))
