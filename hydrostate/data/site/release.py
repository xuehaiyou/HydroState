"""Strict station acceptance and resumable publication of immutable, completed samples."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.data.retrieval.validation import validate_arrays
from hydrostate.datasets.site_dataset import SiteDataset

from .archives import sha256
from .contract import SiteSchema
from .production import audit_selection, prediction_support
from .spatial import static_weights
from .statistics import station_statistics
from .temporal import epoch


def lineage(config, candidates=None):
    split_root = Path(config["split_root"])
    products = json.loads((split_root / "product_support.json").read_text())
    if sha256(config["product_manifest"]) != products["manifest_sha256"]:
        raise ValueError("Product manifest changed since the fixed spatial split")
    if candidates is not None:
        splits = pd.read_parquet(split_root / "station_splits.parquet")
        fixed = splits.groupby("physical_group_id").split.agg(set).to_dict()
        for group, values in candidates.groupby("physical_group_id"):
            if set(values.split) != fixed.get(group):
                raise ValueError("Candidate does not match frozen station split")
        overlaps = pd.read_parquet(split_root / "product_overlaps.parquet")
        if overlaps.site_split.isin(["validation", "test"]).any():
            raise ValueError("Station evaluation overlaps products")
    paths = [
        split_root / n
        for n in (
            "station_splits.parquet",
            "split_config.json",
            "product_support.json",
            "product_overlaps.parquet",
            "joint_product_selection.parquet",
        )
    ]
    paths.extend(sorted(Path(config["footprint_root"]).glob("stations/*/windows.json")))
    return {str(p): sha256(p) for p in paths}


def audit_production(root, plan, *, require_complete=False, require_screened=False):
    root, plan = Path(root), Path(plan)
    config = json.loads((root / "collection.json").read_text())
    report = json.loads((root / "collection_report.json").read_text())
    if require_complete and not report["complete"]:
        raise ValueError("Initial draw and replacements are not complete")
    additive = "success_targets_by_split" in config
    if additive:
        table_counts = pd.read_parquet(root / "samples.parquet").split.value_counts()
        targets_met = all(
            table_counts.get(s, 0) == n for s, n in config["success_targets_by_split"].items()
        )
        if (require_complete or require_screened or report["complete"]) and not targets_met:
            raise ValueError("Additive split success targets not met")
    if require_screened and not additive and not report.get("initial_screening_complete"):
        raise ValueError("Frozen candidate screening has not finished")
    for name, digest in config["plan_sha256"].items():
        if sha256(plan / name) != digest:
            raise ValueError("Frozen sampling plan changed")
    candidates = pd.read_parquet(plan / "input_candidates.parquet")
    if lineage(config, candidates) != config["lineage_sha256"]:
        raise ValueError("Spatial split/footprint lineage changed")
    table = pd.read_parquet(root / "samples.parquet")
    selected = pd.read_parquet(plan / "selected.parquet")
    attempted = {
        p.stem for folder in ("records", "rejected") for p in (root / folder).glob("*.json")
    }
    if (
        not additive
        and (report["complete"] or require_screened)
        and not set(selected.candidate_id).issubset(attempted)
    ):
        raise ValueError("Original selected candidates have not all been attempted")
    audit_selection(table.to_dict("records"), candidates, config["min_date_gap_days"])
    if len(table) != report["successful"] or table.empty:
        raise ValueError("Empty/inconsistent production manifest")
    if len(list((root / "records").glob("*.json"))) != len(table):
        raise ValueError("Manifest does not include every committed sample")
    datasets = {
        split: SiteDataset(root / "samples.parquet", split) for split in table.split.unique()
    }
    product_schema = RetrievalSchema()
    empty_products = {k: np.zeros(s.tail_shape, s.dtype) for k, s in product_schema.targets.items()}
    counts = dict(sm=0, et=0)
    if require_screened:
        atomic_json(
            root / "audit_progress.json", dict(status="running", checked=0, total=len(table))
        )
    for checked, (_, row) in enumerate(table.iterrows(), 1):
        commit = json.loads((root / "records" / f"{row.candidate_id}.json").read_text())
        if commit["sample_id"] != row.sample_id:
            raise ValueError("Manifest/commit identity mismatch")
        dataset = datasets[row.split]
        inputs = dataset._arrays(row, "input", dataset.schema.inputs)
        validate_arrays(inputs, empty_products)
        if inputs["target_bounds"].tolist() != [epoch(row.target_time), epoch(row.target_end)]:
            raise ValueError("Label/input time mismatch")
        schema = SiteSchema(observation_capacity=int(row.observation_capacity))
        labels = dataset._arrays(row, "label", schema.targets)
        dataset._load_targets(row, inputs)
        support = prediction_support(inputs)
        attrs = zarr.open_group(str(root / row.label_path), mode="r").attrs
        observations = attrs["observation_records"]
        indices = dict(sm=0, et=0)
        for item in observations:
            target = item["member"]["target"]
            index = indices[target]
            value = labels[f"site_{target}_native_value"][index]
            weights = labels[f"site_{target}_native_weights"][index]
            if not np.isclose(value, item["label"][target]["value"], rtol=1e-6):
                raise ValueError("Stored label differs from source observation")
            if not support[weights > 0].all():
                raise ValueError("Observation support includes missing input")
            quality = item["spatial_quality"]
            if not quality["accepted"]:
                raise ValueError("Rejected spatial operator was written")
            if target == "sm":
                if weights[20, 20] != 1 or np.count_nonzero(weights) != 1:
                    raise ValueError("SM is not a single centre-pixel observation")
            elif quality["operator"] == "static_210m":
                if not np.array_equal(weights, static_weights()) or any(
                    quality[key] < config[threshold]
                    for key, threshold in (
                        ("homogeneous_fraction", "homogeneous_fraction"),
                        ("valid_fraction", "spatial_valid_fraction"),
                    )
                ):
                    raise ValueError("Static ET violates fixed support or purity")
            elif quality["effective_mass"] < config["footprint_min_mass"]:
                raise ValueError("Footprint absolute mass below threshold")
            indices[target] += 1
            counts[target] += 1
        for target in counts:
            if labels[f"site_{target}_native_valid"].sum() != indices[target]:
                raise ValueError("Observation masks/provenance mismatch")
        if require_screened and checked % 100 == 0:
            atomic_json(
                root / "audit_progress.json",
                dict(status="running", checked=checked, total=len(table)),
            )
    result = dict(
        audit_version="site_full_array_audit_v2",
        status="passed",
        screening_checked=require_screened,
        collection_report_sha256=sha256(root / "collection_report.json"),
        complete=report["complete"],
        samples=len(table),
        observations=counts,
        manifest_sha256=sha256(root / "samples.parquet"),
        collection_sha256=fingerprint(config),
        shortfall=report["shortfall"],
        missing_quotas_are_explicit=True,
    )
    atomic_json(root / "acceptance.json", result)
    if require_screened:
        atomic_json(
            root / "audit_progress.json",
            dict(status="passed", checked=len(table), total=len(table)),
        )
    return result


def publish(root, plan, output, model_validation):
    root, output = Path(root), Path(output)
    if root.resolve() == output.resolve():
        raise ValueError("Publication destination must be independent")
    with locked(root / ".production.lock", blocking=False):
        accepted = audit_production(root, plan, require_complete=True)
        model = json.loads(Path(model_validation).read_text())
        config = json.loads((root / "collection.json").read_text())
        contract_hash = fingerprint(
            {k: v for k, v in config.items() if k not in ("version", "input_profile")}
        )
        if (
            model["status"] != "passed"
            or model["station_statistics"]["data_contract"]["config_sha256"] != contract_hash
        ):
            raise ValueError("Missing model validation for this collection contract")
        table = pd.read_parquet(root / "samples.parquet")
        operators = set(table.et_support) - {"none"}
        if table.valid_sm.gt(0).any():
            operators.add("point")
        if not operators.issubset(model.get("checked_operators", [])):
            raise ValueError("Model pilot did not exercise every published operator type")
        atomic_json(root / "site_target_stats.json", station_statistics(root / "samples.parquet"))
        shutil.copy2(model_validation, root / "training_path.json")
        output.mkdir(parents=True, exist_ok=True)
        with locked(output / ".publish.lock", blocking=False):
            previous = output / "collection.json"
            if previous.exists() and json.loads(previous.read_text()) != config:
                raise ValueError("Publication destination belongs to another collection")
            names = [
                "collection.json",
                "collection_report.json",
                "samples.parquet",
                "acceptance.json",
                "site_target_stats.json",
                "training_path.json",
            ]
            table = pd.read_parquet(root / "samples.parquet")
            for row in table.itertuples():
                names.extend(
                    [row.input_path + "/", row.label_path + "/", f"records/{row.candidate_id}.json"]
                )
            total = sum(p.stat().st_size for p in (root / "zarr").rglob("*") if p.is_file())
            existing = sum(p.stat().st_size for p in (output / "zarr").rglob("*") if p.is_file())
            if shutil.disk_usage(output).free < max(0, total - existing) + 20 * 1024**3:
                raise OSError("SCR capacity insufficient; source samples remain intact")
            with tempfile.NamedTemporaryFile(mode="w") as listing:
                listing.write("\n".join(names) + "\n")
                listing.flush()
                subprocess.run(
                    [
                        "rsync",
                        "-arc",
                        "--files-from=" + listing.name,
                        str(root) + "/",
                        str(output) + "/",
                    ],
                    check=True,
                )
            copied = audit_production(output, plan, require_complete=True)
            if copied != accepted:
                raise ValueError("Published acceptance differs from source")
            # These ID selections preserve product paths relative to their original root.
            shutil.copy2(Path(config["split_root"]) / "joint_product_selection.parquet", output)
            atomic_json(
                output / "_PUBLISHED.json",
                dict(
                    **copied,
                    source=str(root),
                    model_validation=str(model_validation),
                    product_manifest=config["product_manifest"],
                    joint_product_selection="joint_product_selection.parquet",
                ),
            )
            return copied
