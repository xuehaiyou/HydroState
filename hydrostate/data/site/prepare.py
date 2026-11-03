"""Checkpoint window labels from extracted files, before spatial support acceptance."""

import json
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .readers import UnsupportedSource, fluxnet_windows, ismn_windows

LABEL_VERSION = "site_window_labels_am_v1"
READERS = {"ismn": ismn_windows, "fluxnet": fluxnet_windows}


def prepare_labels(paths, output, source, start, end, as_of):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(version=LABEL_VERSION, source=source, start=start, end=end, as_of=as_of)
    config_hash = fingerprint(config)
    summary = dict(config=config, files=[], complete=False)
    with locked(output / ".prepare.lock", blocking=False):
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Window-label configuration changed; use a new output directory")
        atomic_json(config_path, config)
        for index, path in enumerate(paths, 1):
            path = Path(path).resolve()
            stat = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            key = fingerprint(str(path))[:24]
            record_path = output / f"{key}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["source_stat"] != stat or record["config_sha256"] != config_hash:
                    raise ValueError(f"Changed source/config for existing result: {path}")
            else:
                try:
                    rows = READERS[source](path, start, end, as_of)
                    record = dict(status="complete", windows=rows)
                except UnsupportedSource as error:
                    record = dict(status="unsupported", reason=str(error), windows=[])
                # Unexpected parse/I/O errors propagate; they are not empty successes.
                record.update(source_path=str(path), source_stat=stat, config_sha256=config_hash)
                atomic_json(record_path, record)
            counts = dict(
                windows=len(record["windows"]),
                sm=sum(bool((r["sm"] or {}).get("valid")) for r in record["windows"]),
                et_temporal=sum(bool((r["et"] or {}).get("valid")) for r in record["windows"]),
                supervision_ready=sum(r["supervision_ready"] for r in record["windows"]),
            )
            item = dict(
                path=str(record_path),
                source_path=str(path),
                status=record["status"],
                reason=record.get("reason"),
                **counts,
            )
            summary["files"].append(item)
            atomic_json(output / "report.json", summary)
            print(json.dumps(dict(index=index, **item)), flush=True)
        summary["complete"] = True
        atomic_json(output / "report.json", summary)
    return summary
