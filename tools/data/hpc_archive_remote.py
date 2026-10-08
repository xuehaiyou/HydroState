"""Remote half of resumable archive upload (stdlib only)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def safe_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'zarr':
        raise ValueError(f'Unsafe archive path: {relative}')
    destination = root / p
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes destination root')
    return destination


def main():
    request = json.load(sys.stdin)
    root = Path(request['root'])
    transfer = root / '.archive-transfer' / request['snapshot']
    transfer.mkdir(parents=True, exist_ok=True)
    receipt = transfer / (request.get('batch', 'unused') + '.json')
    action = request['action']
    if action == 'receipt':
        print(json.dumps(json.loads(receipt.read_text()) if receipt.exists() else None))
        return
    if action == 'audit':
        missing = []
        for entry in request['entries']:
            p = safe_path(root, entry['path'])
            try:
                s = p.stat()
                matches = (p.is_file() and s.st_size == entry['size']
                           and abs(s.st_mtime - entry['mtime']) < 2)
            except FileNotFoundError:
                matches = False
            if not matches:
                missing.append(entry['path'])
        print(json.dumps(missing))
        return
    if action == 'extract':
        archive = transfer / (request['batch'] + '.tar')
        if digest(archive) != request['archive_sha256']:
            raise ValueError('Archive checksum mismatch')
        expected = {e['path']: e for e in request['entries']}
        seen = set()
        with tarfile.open(archive, 'r:') as bundle:
            for member in bundle:
                if (not member.isfile() or member.name not in expected
                        or member.name in seen or member.size != expected[member.name]['size']):
                    raise ValueError(f'Unexpected archive member: {member.name}')
                seen.add(member.name)
                target = safe_path(root, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.archive-upload')
                with bundle.extractfile(member) as source, open(temporary, 'wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest, 1024 * 1024)
                os.utime(temporary, (member.mtime, member.mtime))
                os.replace(temporary, target)
        if seen != set(expected):
            raise ValueError('Archive is missing expected files')
    elif action != 'complete':
        raise ValueError(action)
    # Receipt is written only after every archive member is installed successfully.
    tmp = receipt.with_suffix('.part')
    tmp.write_text(json.dumps({'manifest_sha256': request['manifest_sha256']}))
    os.replace(tmp, receipt)
    if action == 'extract':
        archive.unlink()
    print(json.dumps({'complete': True}))


if __name__ == '__main__':
    main()
