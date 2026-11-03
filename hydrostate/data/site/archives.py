"""Inventory and permanently extract source archives beside their original ZIPs."""

import hashlib
import json
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath

from hydrostate.data.gee.contracts import atomic_json, locked


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def members(archive):
    """Reject links, path traversal and ambiguous duplicate destinations before writing."""
    seen, result = set(), []
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if (name.is_absolute() or ".." in name.parts or "\\" in info.filename
                or not name.parts or ":" in name.parts[0]
                or name.parts[0] in {"_EXTRACT_OWNER.json", "_EXTRACT_SUCCESS.json"}
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if str(name) in seen:
            raise ValueError(f"Duplicate archive member: {info.filename}")
        seen.add(str(name))
        result.append(info)
    return result


def inventory(paths):
    rows = []
    for raw in paths:
        path = Path(raw).resolve()
        with zipfile.ZipFile(path) as archive:
            entries = members(archive)
            directory = [(i.filename, i.file_size, i.CRC) for i in entries]
        rows.append(dict(
            path=str(path), bytes=path.stat().st_size,
            expanded_bytes=sum(i.file_size for i in entries),
            files=sum(not i.is_dir() for i in entries),
            nested_archives=sum(i.filename.lower().endswith(".zip") for i in entries),
            directory_sha256=hashlib.sha256(json.dumps(directory).encode()).hexdigest(),
        ))
    return rows


def extract(path, *, recursive=True):
    """CRC-checked streaming extraction; completed directories are never unpacked again."""
    path = Path(path).resolve()
    destination = path.with_suffix("")
    marker = destination / "_EXTRACT_SUCCESS.json"
    with locked(str(path) + ".extract.lock"):
        source = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        if marker.exists():
            record = json.loads(marker.read_text())
            if record["source_stat"] != source:
                raise ValueError(f"Source archive changed after extraction: {path}")
        else:
            if destination.exists():
                raise FileExistsError(f"Unowned/incomplete destination: {destination}")
            staging = path.with_name(path.stem + ".extracting")
            owner = staging / "_EXTRACT_OWNER.json"
            if staging.exists():
                if not owner.exists() or json.loads(owner.read_text())["archive"] != str(path):
                    raise FileExistsError(f"Unowned extraction directory: {staging}")
                shutil.rmtree(staging)
            staging.mkdir()
            atomic_json(owner, dict(archive=str(path)))
            started = time.monotonic()
            digest = sha256(path)
            with zipfile.ZipFile(path) as archive:
                entries = members(archive)
                for info in entries:
                    target = staging / info.filename
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as src, target.open("xb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
            record = dict(
                complete=True, archive=str(path), sha256=digest, source_stat=source,
                files=sum(not i.is_dir() for i in entries),
                expanded_bytes=sum(i.file_size for i in entries),
                elapsed_seconds=time.monotonic() - started,
                nested=[i.filename for i in entries if i.filename.lower().endswith(".zip")],
            )
            owner.unlink()
            atomic_json(staging / marker.name, record)
            staging.rename(destination)
    if recursive:
        for nested in record["nested"]:
            extract(destination / nested, recursive=True)
    return dict(record, destination=str(destination))
