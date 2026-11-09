"""Resumable publication of immutable station cohorts and their v3 selection manifests."""

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.joint_dataset import JointRetrievalDataset

from .archives import sha256
from .release import audit_production
from .training_statistics import load_site_statistics

VERSION = "site_joint_release_v1"


def read(path):
    return json.loads(Path(path).read_text())


def require_publication_ready(root):
    """Source working collections remain usable; incomplete publication bundles do not."""
    root = Path(root)
    if not (root / "publication.json").exists():
        return
    marker = root / "_PUBLISHED.json"
    if not marker.exists():
        raise ValueError("Station publication is incomplete; do not train on staging data")
    published = read(marker)
    if published["identity_sha256"] != fingerprint(read(root / "publication.json")):
        raise ValueError("Publication identity changed")
    for name, digest in published["metadata_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Published station metadata changed: {name}")


def full_audit_current(root):
    root = Path(root)
    path = root / "acceptance.json"
    if not path.exists():
        return False
    audit, report = read(path), read(root / "collection_report.json")
    return (
        audit.get("audit_version") == "site_full_array_audit_v2"
        and audit.get("status") == "passed"
        and audit.get("screening_checked") is True
        and audit.get("samples") == report["successful"]
        and audit.get("manifest_sha256") == sha256(root / "samples.parquet")
        and audit.get("collection_sha256") == fingerprint(read(root / "collection.json"))
        and audit.get("collection_report_sha256") == sha256(root / "collection_report.json")
    )


def selection_inputs(split_root, plans, preflight):
    split_root = Path(split_root).resolve()
    report = read(split_root / "split_report.json")
    if (
        report["version"] != "site_frequency_task_spatial_split_v3"
        or report["statistics_status"] != "complete_train_only"
    ):
        raise ValueError("Completed v3 split and training statistics are required")
    stats, _ = load_site_statistics(split_root / "site_target_stats.json")
    check = read(preflight)
    if check.get("status") != "passed" or "actual_mixed_Zarr_batches" not in check.get("scope", ""):
        raise ValueError("A passed real mixed-batch preflight is required")
    for name in ("split_report.json", "site_target_stats.json", "inventory.parquet"):
        if check.get("site_artifacts", {}).get(name) != sha256(split_root / name):
            raise ValueError("Joint preflight does not match this station split")
    if check["statistics"]["site"] != stats:
        raise ValueError("Preflight and station statistics disagree")
    definitions = read(split_root / "datasets.json")["datasets"]
    if len(plans) != len(definitions):
        raise ValueError("Provide one frozen candidate plan per cohort in dataset order")
    roots = [(split_root / d["data_root"]).resolve() for d in definitions]
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate source cohort")
    count = 0
    for root, plan in zip(roots, plans):
        production = read(root / "collection.json")
        progress = read(root / "collection_report.json")
        if not progress.get("initial_screening_complete"):
            raise ValueError("Source cohort is still collecting")
        if sha256(root / "samples.parquet") != report["source_manifests"][str(root)]:
            raise ValueError("Source manifest changed after resplitting")
        for name, digest in production["plan_sha256"].items():
            if sha256(Path(plan) / name) != digest:
                raise ValueError("Frozen candidate plan changed")
        count += progress["successful"]
    if count != report["samples"]:
        raise ValueError("Split count does not match completed cohorts")
    product = read(roots[0] / "collection.json")["product_manifest"]
    # Recheck selections, isolation and training fingerprints, not just a historical success flag.
    for split in ("train", "validation", "test"):
        dataset = JointRetrievalDataset(product, split_root, split)
        dataset.validate_site_statistics(stats)
    return split_root, report, definitions, roots, product


def cohort_files(root):
    names = [
        "collection.json",
        "collection_report.json",
        "samples.parquet",
        "records/",
        "rejected/",
    ]
    table = pd.read_parquet(root / "samples.parquet")
    for field in ("input_path", "label_path"):
        for value in table[field]:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or path == Path("."):
                raise ValueError("Zarr paths must stay inside their cohort")
            if not (root / path).is_dir():
                raise FileNotFoundError(root / path)
            names.append(str(path) + "/")
    return sorted(set(names))


def walk_files(root, names):
    """Measure the selected payload once; caches and unrelated source files are excluded."""
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ValueError("Publication sources must be regular files/directories")
        if path.is_file():
            yield path
        elif path.is_dir():
            for folder, directories, files in os.walk(path):
                if any((Path(folder) / d).is_symlink() for d in directories):
                    raise ValueError("Symlink in station Zarr")
                for filename in files:
                    item = Path(folder) / filename
                    if item.is_symlink():
                        raise ValueError("Symlink in station Zarr")
                    yield item
        else:
            raise FileNotFoundError(path)


def capacity(roots, names, output, reserve_gib):
    if reserve_gib < 0:
        raise ValueError("Negative free-space reserve")
    total = remaining = count = 0
    for i, (root, files) in enumerate(zip(roots, names)):
        for path in walk_files(root, files):
            size = path.stat().st_size
            target = output / f"cohorts/{i}" / path.relative_to(root)
            existing = target.stat().st_size if target.is_file() else 0
            total += size
            remaining += max(0, size - existing)
            count += 1
    parent = output
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    # Also covers small manifest/proof copies and rsync's temporary replacement files.
    required = remaining + int(reserve_gib * 1024**3)
    return dict(
        payload_bytes=total,
        payload_files=count,
        remaining_bytes=remaining,
        free_bytes=free,
        required_bytes=required,
        enough_space=free >= required,
    )


def copy_files(source, target, names):
    """Resume using rsync size/mtime checks; require a successful copy exit status."""
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w") as listing:
        listing.write("\n".join(names) + "\n")
        listing.flush()
        command = [
            "rsync",
            "-rlt",
            "--files-from=" + listing.name,
            str(source) + "/",
            str(target) + "/",
        ]
        subprocess.run(command, check=True)



def verify_relocated(output, product, stats):
    counts = {}
    for split in ("train", "validation", "test"):
        dataset = JointRetrievalDataset(product, output, split)
        dataset.validate_site_statistics(stats)
        indices = [0]
        for i, site in enumerate(dataset.sites, 1):
            for mask in (
                site.table.valid_sm.gt(0),
                site.table.et_support.eq("static_210m"),
                site.table.et_support.eq("footprint"),
            ):
                if mask.any():
                    indices.append(dataset.offsets[i] + int(mask[mask].index[0]))
        hydro_collate([dataset[i] for i in dict.fromkeys(indices)])
        counts[split] = sum(len(d) for d in dataset.sites)
    return counts


def publish_joint(
    split_root,
    plans,
    output,
    preflight,
    *,
    code_commit,
    reserve_gib=20,
    check_only=False,
    skip_capacity_check=False,
):
    output = Path(output).resolve()
    split_root, report, definitions, roots, product = selection_inputs(split_root, plans, preflight)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in [split_root, *roots]):
        raise ValueError("Publication must use an independent destination")
    source_names = [
        "split_report.json",
        "site_target_stats.json",
        "inventory.parquet",
        "datasets.json",
        "station_splits.parquet",
    ]
    for item in definitions:
        path = (split_root / item["manifest"]).resolve().relative_to(split_root)
        source_names.append(str(path))
    identity = dict(
        version=VERSION,
        code_commit=code_commit,
        source=str(split_root),
        source_sha256={n: sha256(split_root / n) for n in source_names},
        preflight_sha256=sha256(preflight),
        collections={str(r): sha256(r / "collection.json") for r in roots},
        plans=[str(Path(p).resolve()) for p in plans],
    )
    if (output / "publication.json").exists():
        if read(output / "publication.json") != identity:
            raise ValueError("Destination belongs to another publication")
        if (output / "_PUBLISHED.json").exists():
            require_publication_ready(output)
            return read(output / "_PUBLISHED.json")
    elif output.exists() and any(p.name != ".publish.lock" for p in output.iterdir()):
        raise ValueError("Refusing to overwrite an unrelated destination")
    names = [cohort_files(root) for root in roots]
    space = (
        dict(status="skipped_by_user", enough_space=None)
        if skip_capacity_check
        else capacity(roots, names, output, reserve_gib)
    )
    pending = [str(r) for r in roots if not full_audit_current(r)]
    if check_only:
        return dict(
            status="checked",
            space=space,
            pending_full_audits=pending,
            samples=report["samples"],
            output=str(output),
            published=False,
        )
    if not skip_capacity_check and not space["enough_space"]:
        raise OSError(f"Insufficient destination capacity: {space}")
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(locked(output / ".publish.lock", blocking=False))
        for root in sorted(roots):
            stack.enter_context(locked(root / ".production.lock", blocking=False))
        # Check again after locking, before creating or updating any payload.
        selection_inputs(split_root, plans, preflight)
        if (output / "publication.json").exists() and read(output / "publication.json") != identity:
            raise ValueError("Destination identity changed")
        if (output / "_PUBLISHED.json").exists():
            require_publication_ready(output)
            return read(output / "_PUBLISHED.json")
        atomic_json(output / "publication.json", identity)
        for i, (root, plan, files) in enumerate(zip(roots, plans, names)):
            atomic_json(output / "publication_progress.json", dict(stage="auditing", cohort=i))
            if not full_audit_current(root):
                audit_production(root, plan, require_screened=True)
            if not full_audit_current(root):
                raise ValueError("Full cohort audit is not current")
            atomic_json(output / "publication_progress.json", dict(stage="copying", cohort=i))
            copy_files(root, output / f"cohorts/{i}", files + ["acceptance.json"])
            copy_files(
                Path(plan),
                output / f"plans/{i}",
                list(read(root / "collection.json")["plan_sha256"]),
            )
        proof = split_root / report["joint_product_selection"]
        proof_names = [proof.name, "station_splits.parquet", "product_support.json"]
        proof_names += [
            n
            for n in ("split_config.json", "product_overlaps.parquet")
            if (proof.parent / n).exists()
        ]
        copy_files(proof.parent, output / "proofs", proof_names)
        copy_files(split_root, output / "provenance", source_names)
        shutil.copy2(preflight, output / "provenance/source_preflight.json")
        for name in source_names:
            if name not in ("split_report.json", "datasets.json"):
                (output / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(split_root / name, output / name)
        relocated = []
        for i, item in enumerate(definitions):
            manifest = (split_root / item["manifest"]).resolve().relative_to(split_root)
            relocated.append(
                dict(type="SiteDataset", manifest=str(manifest), data_root=f"cohorts/{i}")
            )
        atomic_json(output / "datasets.json", dict(datasets=relocated))
        published_report = dict(
            report,
            published=False,
            source_manifests={
                f"cohorts/{i}": report["source_manifests"][str(r)] for i, r in enumerate(roots)
            },
            joint_product_selection=f"proofs/{proof.name}",
        )
        atomic_json(output / "split_report.json", published_report)
        atomic_json(output / "publication_progress.json", dict(stage="verifying_relocated_readers"))
        counts = verify_relocated(output, product, read(output / "site_target_stats.json"))
        if sum(counts.values()) != report["samples"]:
            raise ValueError("Relocated split count mismatch")
        # Freeze provenance and selections again before publishing the final success marker.
        if any(sha256(split_root / n) != digest for n, digest in identity["source_sha256"].items()):
            raise ValueError("Source split changed during publication")
        published_report["published"] = True
        atomic_json(output / "split_report.json", published_report)
        metadata = list(source_names) + [f"proofs/{n}" for n in proof_names]
        metadata += [f"provenance/{n}" for n in source_names]
        metadata.append("provenance/source_preflight.json")
        for i in range(len(roots)):
            metadata += [
                f"cohorts/{i}/{n}"
                for n in (
                    "collection.json",
                    "collection_report.json",
                    "samples.parquet",
                    "acceptance.json",
                )
            ]
            metadata += [
                f"plans/{i}/{n}" for n in read(roots[i] / "collection.json")["plan_sha256"]
            ]
        result = dict(
            status="passed",
            version=VERSION,
            identity_sha256=fingerprint(identity),
            samples=report["samples"],
            by_split=counts,
            source=str(split_root),
            metadata_sha256={n: sha256(output / n) for n in metadata},
            verification="full_source_array_audits+rsync_copy_exit_status+relocated_mixed_batches",
            model_training_validated=False,
            space=space,
        )
        atomic_json(output / "_PUBLISHED.json", result)
        atomic_json(
            output / "publication_progress.json", dict(stage="published", samples=report["samples"])
        )
        return result
