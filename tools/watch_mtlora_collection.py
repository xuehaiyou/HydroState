"""Wait without GPUs, then prepare and submit the authorized MTLoRA/LoRA pair.

Run from a frozen release under nohup on the login host. One flock owner; durable
submission intents prevent duplicate jobs after interruption. Errors stop the
watcher for inspection; it never cancels jobs or silently changes input data.
"""
import argparse
import fcntl
import getpass
import json
import os
import re
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def save(path, record):
    record['checked_utc'] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def submit(path, record, stage, script, env, *extra):
    previous = record['jobs'].get(stage)
    if previous:
        if previous.get('job_id'):
            return previous['job_id']
        raise RuntimeError(f'Inspect ambiguous or failed submission before retry: {stage}')
    root = path.parent
    command = ['sbatch', '--parsable', f'--chdir={record["code_root"]}',
               f'--job-name=hs_mtlora_{stage}_{root.name}', '--export=ALL',
               f'--output={root}/logs/{stage}-%j.out',
               f'--error={root}/logs/{stage}-%j.err', script, *extra]
    entry = dict(command=command, status='submitting')
    record['jobs'][stage] = entry
    save(path, record)
    response = subprocess.run(command, env=env, capture_output=True, text=True)
    entry.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
    save(path, record)
    response.check_returncode()
    if not re.fullmatch(r'\d+(;[A-Za-z0-9_.-]+)?', response.stdout.strip()):
        raise RuntimeError('Ambiguous sbatch response: inspect Slurm before restarting')
    job, _, cluster = response.stdout.strip().partition(';')
    entry.update(job_id=job, cluster=cluster or None, status='submitted')
    save(path, record)
    print(stage, job, flush=True)
    return job


def free_gpu_slots():
    limits = subprocess.check_output([
        'sacctmgr', '-nP', 'show', 'qos', 'gpu',
        'format=MaxSubmitJobsPA,MaxSubmitJobsPU'], text=True).strip().split('|')
    rows = subprocess.check_output([
        'squeue', '--array', '--noheader', '--account=geog_geors', '--qos=gpu',
        '--format=%i|%u'], text=True).splitlines()
    counts = [len(rows), sum(line.split('|')[1] == getpass.getuser() for line in rows)]
    return min([2] + [int(limit) - count for limit, count in zip(limits[:2], counts) if limit])


def advance(path, record):
    root = path.parent
    data = Path(record['data_root'])
    if not (data / '_COLLECTION_SUCCESS.json').exists():
        record['status'] = 'waiting_for_collection'
        return False
    env = {k: v for k, v in os.environ.items() if not k.startswith('HYDROSTATE_')}
    env.update(record['environment'])
    job = submit(path, record, 'prepare', 'scripts/slurm/prepare_retrieval.sh', env)
    accounting = subprocess.check_output([
        'sacct', '-nP', '-X', '-j', job, '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [row.split('|') for row in accounting.splitlines() if row.startswith(job + '|')]
    state = rows[0][1] if rows else 'ACCOUNTING_PENDING'
    record['prepare_state'] = state
    if state != 'COMPLETED':
        if state.split()[0].rstrip('+') in {
            'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
            'BOOT_FAIL', 'DEADLINE', 'REVOKED'}:
            raise RuntimeError(f'Preparation {job} ended with {state}')
        record['status'] = 'waiting_for_preparation'
        return False
    if rows[0][2] != '0:0' or not (root / 'provenance/ready.json').exists():
        raise RuntimeError('Preparation did not publish its validated ready record')
    for stage in ('mtlora', 'lora'):
        if not record['jobs'].get(stage, {}).get('job_id') and free_gpu_slots() <= 0:
            record['status'] = 'waiting_for_gpu_submission_capacity'
            return False
        stage_env = dict(env, HYDROSTATE_WORK_DIR=str(root / stage))
        submit(path, record, stage, 'scripts/slurm/train_retrieval.sh', stage_env, stage)
    record['status'] = 'both_training_jobs_submitted'
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('Polling interval must be at least 30 seconds')
    with args.record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(args.record.read_text())
        os.chdir(record['code_root'])
        subprocess.run(['sha256sum', '--check', '--status',
                        record['environment']['HYDROSTATE_SOURCE_HASHES']], check=True)
        record.update(watcher_pid=os.getpid(), watcher_host=socket.gethostname())
        try:
            while True:
                done = advance(args.record, record)
                save(args.record, record)
                print(record['checked_utc'], record['status'], flush=True)
                if done:
                    break
                time.sleep(args.interval)
        except Exception as error:
            record.update(status='stopped_on_error', error=repr(error))
            save(args.record, record)
            raise


if __name__ == '__main__':
    main()
