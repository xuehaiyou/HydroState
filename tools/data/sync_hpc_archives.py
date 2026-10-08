#!/usr/bin/env python3
"""Resume existing Zarr uploads using small batches of uncompressed tar archives."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--shards-per-batch', type=int, default=500)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    lock = (snapshot / ".archive-upload.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = args.source.resolve()
    name = snapshot.name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid snapshot name')
    if args.shards_per_batch < 1:
        raise ValueError('Invalid batch size')
    remote = os.environ.get('HPC_HOST', 'xiaoz@hpc2021.hku.hk')
    root = os.environ.get('HPC_DATA_ROOT', '/fossfs/xiaozhen/HydroState/samples_v1')
    python = os.environ.get('HPC_PYTHON', '/home/xiaoz/miniconda/envs/hydrostate/bin/python')
    code = os.environ.get('HPC_CODE_ROOT', '/home/xiaoz/HydroState')
    for value in (root, python, code):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value):
            raise ValueError('Unsupported remote path')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
           '-o', 'StrictHostKeyChecking=yes']
    known = '/mnt/c/Users/zhenxiao/.ssh/known_hosts'
    if Path(known).exists():
        ssh += ['-o', 'UserKnownHostsFile=' + known]
    key = '/home/xiaoz/.ssh/id_rsa'
    if Path(key).exists():
        ssh += ['-o', 'IdentitiesOnly=yes', '-i', key]
    helper = Path(__file__).with_name('hpc_archive_remote.py').read_text()
    manifest_hash = sha(snapshot / 'samples.parquet')
    # Include partitioning and shard-list identity in receipt namespace.
    run_id = hashlib.sha256((manifest_hash + sha(snapshot / 'zarr-files.txt')
                            + str(args.shards_per_batch)).encode()).hexdigest()[:16]
    transfer_name = name + '-' + run_id
    def rpc(action, batch, **kwargs):
        request = dict(action=action, root=root, snapshot=transfer_name,
                       batch=batch, manifest_sha256=manifest_hash, **kwargs)
        result = subprocess.run(ssh + [remote, shlex.quote(python) + ' -c ' + shlex.quote(helper)],
                                input=json.dumps(request), text=True,
                                stdout=subprocess.PIPE, check=True)
        return json.loads(result.stdout)
    def upload(local, destination):
        subprocess.run(['rsync', '-t', '--partial', '--partial-dir=.rsync-partial',
                        '--info=progress2', '-e', shlex.join(ssh), str(local),
                        remote + ':' + destination], check=True)
    remote_snapshot = root + '/manifests/' + name
    result = subprocess.run(ssh + [remote, f"if test -f '{remote_snapshot}/samples.parquet'; "
                        f"then sha256sum '{remote_snapshot}/samples.parquet'; fi"],
                        text=True, stdout=subprocess.PIPE, check=True)
    if result.stdout.strip():
        if result.stdout.split()[0] != manifest_hash:
            raise ValueError('Remote snapshot differs; choose another name')
        print('Snapshot already published and unchanged.', flush=True)
        return
    shards = (snapshot / 'zarr-files.txt').read_text().splitlines()
    cache = snapshot / ('archive-cache-' + run_id)
    cache.mkdir(exist_ok=True)
    batches = (len(shards) + args.shards_per_batch - 1) // args.shards_per_batch
    for offset in range(0, len(shards), args.shards_per_batch):
        number = offset // args.shards_per_batch
        batch = f'{number:05d}'
        receipt = rpc('receipt', batch)
        if receipt and receipt['manifest_sha256'] == manifest_hash:
            print(f'Batch {number + 1}/{batches}: already complete', flush=True)
            continue
        archive = cache / (batch + '.tar')
        plan = cache / (batch + '.json')
        if archive.exists() and plan.exists():
            payload = json.loads(plan.read_text())
            if sha(archive) != payload['archive_sha256']:
                raise ValueError('Local archive changed; inspect cache before retrying')
        else:
            entries = []
            for relative in shards[offset:offset + args.shards_per_batch]:
                p = Path(relative)
                if p.is_absolute() or '..' in p.parts or p.parts[0] != 'zarr':
                    raise ValueError('Unsafe shard path')
                if not (source / p / '_SUCCESS.json').is_file():
                    raise ValueError(f'Incomplete source shard: {p}')
                for directory, _, filenames in os.walk(source / p):
                    for filename in sorted(filenames):
                        f = Path(directory) / filename
                        stat = f.stat()
                        entries.append(dict(path=f.relative_to(source).as_posix(),
                                            size=stat.st_size, mtime=stat.st_mtime))
            missing = set(rpc('audit', batch, entries=entries))
            selected = [e for e in entries if e['path'] in missing]
            print(f'Batch {number + 1}/{batches}: {len(entries)-len(selected)} files already '
                  f'present; {len(selected)} remaining', flush=True)
            if not selected:
                rpc('complete', batch)
                continue
            partial = archive.with_suffix('.tar.part')
            with tarfile.open(partial, 'w', format=tarfile.PAX_FORMAT) as bundle:
                for entry in selected:
                    bundle.add(source / entry['path'], arcname=entry['path'], recursive=False)
            os.replace(partial, archive)
            payload = dict(entries=selected, archive_sha256=sha(archive))
            plan.write_text(json.dumps(payload))
        destination = root + '/.archive-transfer/' + transfer_name + '/' + batch + '.tar'
        upload(archive, destination)
        rpc('extract', batch, **payload)
        archive.unlink()
        plan.unlink()
        print(f'Batch {number + 1}/{batches}: uploaded and extracted', flush=True)
    subprocess.run(ssh + [remote, f"mkdir -p '{remote_snapshot}'"], check=True)
    upload(snapshot / 'samples.parquet', remote_snapshot + '/samples.parquet.upload')
    upload(snapshot / 'snapshot.json', remote_snapshot + '/snapshot.json')
    subprocess.run(ssh + [remote, f"cd '{code}' && '{python}' tools/validate_zarr.py "
                    f"'{remote_snapshot}/samples.parquet.upload' --data-root '{root}' && "
                    f"mv '{remote_snapshot}/samples.parquet.upload' '{remote_snapshot}/samples.parquet'"],
                   check=True)
    print('Published: ' + remote_snapshot + '/samples.parquet', flush=True)


if __name__ == '__main__':
    main()
