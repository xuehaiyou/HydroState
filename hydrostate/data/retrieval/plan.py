"""Immutable sample provenance, mutable collection targets, and frozen releases."""

import ast
import json
import shutil
import signal
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, current_thread, main_thread

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.storage.packs import file_hash

from .contract import PROFILE, VERSION

# These affect execution and sampling effort, not an already produced shard.
RUN_FIELDS = {
    "samples",
    "concurrency",
    "candidate_block_batch",
    "candidate_pool_factor",
    "candidate_pool_capacity",
    "candidate_low_watermark",
    "ice_candidate_buffer",
    "ice_candidate_low_watermark",
    "producer_hash",
    "candidate_dir",
    "project",
}
DATA_MODULES = (
    "hydrostate/data/retrieval/alignment.py",
    "hydrostate/data/retrieval/composites.py",
    "hydrostate/data/retrieval/contract.py",
    "hydrostate/data/retrieval/gee_labels.py",
    "hydrostate/data/retrieval/labels.py",
    "hydrostate/data/retrieval/specs.py",
    "hydrostate/data/retrieval/validation.py",
    "hydrostate/data/native_support.py",
    "hydrostate/data/spatial.py",
    "hydrostate/data/gee/products.py",
    "hydrostate/data/gee/client.py",
    "hydrostate/data/gee/contracts.py",
    "hydrostate/data/gee/backends/pixels.py",
    "hydrostate/storage/writer.py",
    "hydrostate/storage/schema.py",
)
DATA_FUNCTIONS = {
    "hydrostate/data/retrieval/collector.py": ("epoch", "input_arrays", "collect_sample"),
    "hydrostate/data/retrieval/catalog.py": (
        "cached_json",
        "pml_calendar",
        "catalog",
        "centre_sample",
    ),
}


def data_signature(repo):
    """Hash pixel-producing dependencies independently of the scheduling code.

    Two mixed modules contribute their data functions and the imports those
    functions use. Function-local imports are already included in the AST.
    """
    repo = Path(repo)
    parts = {p: file_hash(repo / p) for p in DATA_MODULES}
    for path, names in DATA_FUNCTIONS.items():
        module = ast.parse((repo / path).read_text())
        definitions = [
            n
            for n in module.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        if {n.name for n in definitions} != set(names):
            raise ValueError(f"Missing data producer functions in {path}")
        used = {n.id for node in definitions for n in ast.walk(node) if isinstance(n, ast.Name)}
        imports = []
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                aliases = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used]
                if aliases:
                    node.names = aliases
                    imports.append(node)
        parts[path] = [ast.dump(n, include_attributes=False) for n in [*imports, *definitions]]
    return fingerprint(parts)


def legacy_producer_hash(repo):
    repo = Path(repo)
    sources = [
        *sorted((repo / "hydrostate/data/retrieval").glob("*.py")),
        *(
            repo / p
            for p in (
                "hydrostate/data/native_support.py",
                "hydrostate/data/gee/products.py",
                "hydrostate/data/spatial.py",
                "hydrostate/storage/writer.py",
                "hydrostate/storage/schema.py",
            )
        ),
    ]
    return fingerprint({str(p.relative_to(repo)): file_hash(p) for p in sources})


def target_samples(root, config):
    path = Path(root) / "collection_plan.json"
    return json.loads(path.read_text())["samples"] if path.exists() else config["samples"]


def snapshot_release(root):
    """Copy only small manifests/metadata; immutable Zarr files stay in place."""
    root = Path(root)
    marker = root / "_COLLECTION_SUCCESS.json"
    if not marker.exists():
        return
    report = json.loads(marker.read_text())
    destination = root / "releases" / f"n{report['samples']}_{report['manifest_sha256'][:12]}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "samples.parquet",
        "collection.json",
        "collection_plan.json",
        "collection_producer.json",
        "collection_report.json",
        "_COLLECTION_SUCCESS.json",
    ):
        source = root / name
        if source.exists() and not (destination / name).exists():
            shutil.copy2(source, destination / name)
    for source in root.glob("*stats*.json"):
        if not (destination / source.name).exists():
            shutil.copy2(source, destination / source.name)


def bind_plan(root, config, *, adopt_from=None):
    """Called under the collection lock; existing shard identities never change."""
    root, repo = Path(root), Path(__file__).resolve().parents[3]
    path = root / "collection.json"
    identity = dict(config, version=VERSION, input_profile=PROFILE)
    if not path.exists():
        if any(p.name != ".collection.lock" for p in root.iterdir()):
            raise ValueError("Refusing to adopt an existing dataset")
        atomic_json(path, identity)
    original = json.loads(path.read_text())

    def fixed(c):
        return {k: v for k, v in c.items() if k not in RUN_FIELDS}

    if fixed(original) != fixed(identity):
        raise ValueError("Collection contract changed; use a new output directory")
    original_config = {k: v for k, v in original.items() if k not in ("version", "input_profile")}
    signature = data_signature(repo)
    provenance_path = root / "collection_producer.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["data_producer_hash"] != signature or provenance[
            "config_sha256"
        ] != fingerprint(original_config):
            raise ValueError("Data producer or immutable collection configuration changed")
    else:
        if original["producer_hash"] != config["producer_hash"]:
            if adopt_from is None:
                raise ValueError(
                    "Scheduler migration requires --adopt-from with the previous frozen code"
                )
            if (
                legacy_producer_hash(adopt_from) != original["producer_hash"]
                or data_signature(adopt_from) != signature
            ):
                raise ValueError("Previous producer does not match; refusing scheduler migration")
        atomic_json(
            provenance_path,
            dict(
                data_producer_hash=signature,
                config_sha256=fingerprint(original_config),
                adopted_from=str(adopt_from) if adopt_from else None,
            ),
        )
    previous = target_samples(root, original)
    if config["samples"] < previous:
        raise ValueError("Target sample count may only increase")
    if config["samples"] > previous:
        snapshot_release(root)
        (root / "_COLLECTION_SUCCESS.json").unlink(missing_ok=True)
    plan = {k: config[k] for k in RUN_FIELDS if k in config}
    plan["updated_at"] = datetime.now(UTC).isoformat()
    plan["data_producer_hash"] = signature
    atomic_json(root / "collection_plan.json", plan)
    return original_config


@contextmanager
def collection_stop():
    stop, previous = Event(), {}
    if current_thread() is main_thread():
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, lambda *_: stop.set())
    try:
        yield stop
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
