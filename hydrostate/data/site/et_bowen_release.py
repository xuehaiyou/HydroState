"""Pack one ET Zarr, replace active ET selections, and retain immutable SM cohorts."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.joint_dataset import JointRetrievalDataset
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.normalization import training_fingerprint
from hydrostate.storage import SampleShardWriter

from .active import active_release
from .archives import sha256
from .contract import SiteSchema
from .resplit import stratify_groups, task_coverage
from .statistics import station_statistics

RELEASE_VERSION = "site_task_specific_et_replacement_v4"


def assign_et_splits(table, base, root):
    known = pd.read_parquet(base / "splits_20261103/station_splits.parquet")
    locked = set(known.loc[known.product_overlap_count.fillna(0).gt(0), "spatial_group_id"])
    previous = set(json.loads((root / "plan/previous_et_training.json").read_text()))
    locked.update(table.loc[table.physical_group_id.isin(previous), "spatial_group_id"])
    assignments = stratify_groups(table, locked, seed=20261110)
    table = table.copy()
    table["split"] = table.spatial_group_id.map(assignments)
    if table.loc[table.spatial_group_id.isin(locked), "split"].ne("train").any():
        raise ValueError("ET product/development group entered evaluation")
    return table, locked


def pack(root, table):
    """Single root group with native input arrays, ET labels and per-sample quality."""
    output = root / "release"
    if (output / "acceptance.json").exists():
        return output
    output.mkdir(exist_ok=True)
    if (output / "et.zarr").exists():
        shutil.rmtree(output / "et.zarr")  # Unaccepted packing output only.
    shutil.copy2(root / "collection.json", output / "collection.json")
    source = SiteDataset(root / "samples.parquet", "train", data_root=root)
    attrs = dict(
        config_sha256=source.config_hash,
        input_profile=PROFILE,
        normalization_hash=json.loads((root / "collection.json").read_text())["normalization_hash"],
        grid_metadata_storage="per_sample_json",
        label_definition="provider_Bowen_corrected_ET",
    )
    temporary = output / "et.partial.zarr"
    if temporary.exists():
        shutil.rmtree(temporary)  # Only this uncommitted packing output is rebuilt on resume.
    temporary.mkdir()
    group = zarr.open_group(str(temporary), mode="w")
    writer = SampleShardWriter(
        str(temporary / "inputs"), str(temporary / "labels"), len(table), SiteSchema(), attrs=attrs
    )
    grids, metadata, scenes = [], [], []
    table = table.copy().reset_index(drop=True)
    for index, row in table.iterrows():
        inputs = source._arrays(row, "input", source.schema.inputs)
        labels = source._arrays(row, "label", SiteSchema().targets)
        if labels["site_sm_native_valid"].any() or labels["site_et_native_valid"].sum() != 1:
            raise ValueError("ET release must contain exactly one ET observation and no SM")
        if not np.isclose(labels["site_et_native_value"][0], row.et_corrected, rtol=1e-6):
            raise ValueError("Corrected label/manifest mismatch")
        if not np.isclose(labels["site_et_native_weights"].sum(), 1, atol=1e-5):
            raise ValueError("ET spatial operator is not normalized")
        old = zarr.open_group(str(root / row.input_path), mode="r").attrs
        label = zarr.open_group(str(root / row.label_path), mode="r").attrs
        grids.append(json.dumps(old["input_grids"]))
        metadata.append(json.dumps(label["observation_records"]))
        scenes.append(json.dumps(old["scenes"]))
        writer.write(inputs, labels)
    writer.finalize()
    inputs = zarr.open_group(str(temporary / "inputs"), mode="a")
    for name, values in [("input_grid_metadata", grids), ("scene_metadata", scenes)]:
        inputs.create_dataset(name, data=np.asarray(values, dtype=str), chunks=(1,))
    group.create_dataset("observation_metadata", data=np.asarray(metadata, dtype=str), chunks=(1,))
    quality = group.create_group("quality")
    for name in [
        "et_corrected",
        "et_uncorrected",
        "homogeneous_fraction",
        "classification_valid_fraction",
        "et_valid_fraction",
        "et_measured_fraction",
        "et_day_fraction",
        "et_night_fraction",
    ]:
        quality.create_dataset(
            name, data=table[name].to_numpy(dtype="float32"), chunks=(min(256, len(table)),)
        )
    for name in ["sample_id", "physical_group_id", "target_time", "target_end", "split"]:
        quality.create_dataset(name, data=table[name].astype(str).to_numpy(dtype=str), chunks=(1,))
    group.attrs.update(
        complete=True,
        version=RELEASE_VERSION,
        task="et",
        samples=len(table),
        homogeneous_fraction_threshold=0.90,
        classification_coverage_threshold=0.90,
    )
    os.replace(temporary, output / "et.zarr")
    table["input_path"] = "et.zarr/inputs"
    table["label_path"] = "et.zarr/labels"
    table["shard_index"] = np.arange(len(table))
    table.to_parquet(output / "samples.parquet", index=False)
    # Actual reads cover each split/operator; no second full-array or checksum traversal.
    checked = []
    for split in table.split.unique():
        dataset = SiteDataset(output / "samples.parquet", split)
        for operator, rows in dataset.table.groupby("et_support"):
            item = dataset[int(rows.index[0])]
            if item["data_samples"]["site_et_native_valid"].sum() != 1:
                raise ValueError("Packed ET dataset mask mismatch")
            checked.append(dict(split=split, operator=operator))
    atomic_json(output / "site_target_stats.json", station_statistics(output / "samples.parquet"))
    atomic_json(
        output / "acceptance.json",
        dict(
            status="passed",
            samples=len(table),
            checked=checked,
            validation="all labels checked while packing; representative actual reads",
            by_bin=table.frequency_bin.value_counts().to_dict(),
            by_split=table.split.value_counts().to_dict(),
        ),
    )
    return output


def combine_statistics(parts):
    result = {}
    for name in ("site_sm", "site_et"):
        values = [p["observations"][name] for p in parts if name in p["observations"]]
        if not values:
            continue
        n = sum(v["count"] for v in values)
        mean = sum(v["mean"] * v["count"] for v in values) / n
        variance = sum(v["count"] * (v["std"] ** 2 + (v["mean"] - mean) ** 2) for v in values) / n
        result[name] = dict(count=n, mean=mean, std=variance**0.5)
    return dict(
        version="site_training_statistics_v2",
        split="train",
        ddof=0,
        observations=result,
        cohorts=parts,
    )


def stage_release(base, root, scr, et_root, locked):
    """SM source data and split assignments stay intact; only ET is replaced."""
    old = active_release(scr)
    previous = root / "plan/previous_active_release.json"
    expected = Path(json.loads(previous.read_text())) if previous.exists() else scr
    if old.resolve() != expected.resolve():
        raise ValueError("An unexpected newer release is already active")
    dest = scr / "releases" / root.name
    dest.mkdir(parents=True, exist_ok=True)
    definitions = json.loads((old / "datasets.json").read_text())["datasets"]
    previous_stats = json.loads((old / "site_target_stats.json").read_text())
    previous_report = json.loads((old / "split_report.json").read_text())
    frames, parts, newdefs = [], [], []
    for i, item in enumerate(definitions):
        manifest = old / item["manifest"]
        source = (old / item["data_root"]).resolve()
        original = pd.read_parquet(manifest)
        if (original.valid_sm.gt(0) & original.valid_et.gt(0)).any():
            raise ValueError(
                "Mixed-task source rows need explicit mask migration before replacement"
            )
        sm = original[original.valid_sm.gt(0)].copy()
        if sm.empty:
            continue
        sm["cohort"] = len(newdefs)
        path = dest / f"cohort_{len(newdefs)}.parquet"
        sm.to_parquet(path, index=False)
        dataset = SiteDataset(path, "train", data_root=source)
        part = dict(previous_stats["cohorts"][i])
        # Every original SM row/split is preserved, so its moments remain exactly valid.
        if part["observations"]["site_sm"]["count"] != int(
            sm.loc[sm.split.eq("train"), "valid_sm"].sum()
        ):
            raise ValueError("SM training population changed; cannot reuse old moments")
        part["observations"] = {"site_sm": part["observations"]["site_sm"]}
        part["training_fingerprint"] = training_fingerprint(sm, dataset.data_contract)
        parts.append(part)
        frames.append(sm)
        newdefs.append(
            dict(type="SiteDataset", manifest=path.name, data_root=str(source), task="sm")
        )
    et = pd.read_parquet(et_root / "samples.parquet")
    et["cohort"] = len(newdefs)
    path = dest / f"cohort_{len(newdefs)}.parquet"
    et.to_parquet(path, index=False)
    parts.append(station_statistics(path, data_root=et_root))
    frames.append(et)
    newdefs.append(dict(type="SiteDataset", manifest=path.name, data_root=str(et_root), task="et"))
    inventory = pd.concat(frames, ignore_index=True)
    inventory.to_parquet(dest / "inventory.parquet", index=False)
    atomic_json(dest / "datasets.json", dict(datasets=newdefs))
    atomic_json(dest / "site_target_stats.json", combine_statistics(parts))
    proof = (old / previous_report["joint_product_selection"]).resolve()
    coverage = [r for r in previous_report["task_coverage"] if r["task"] == "sm"]
    coverage += [r for r in task_coverage(et, locked) if r["task"] == "et"]
    report = dict(
        version=RELEASE_VERSION,
        samples=len(inventory),
        task_coverage=coverage,
        cross_task_colocation="allowed; not globally unseen-location evaluation",
        source_manifests={
            i["data_root"]: sha256(Path(i["data_root"]) / "samples.parquet") for i in newdefs
        },
        source_splits_sha256=sha256(proof.parent / "station_splits.parquet"),
        joint_product_selection=str(proof),
        joint_product_selection_sha256=sha256(proof),
        statistics_status="complete_train_only",
        published=False,
        replaced_et=True,
        sm_unchanged=True,
        previous_release=str(old),
    )
    atomic_json(dest / "split_report.json", report)
    product = json.loads((proof.parent / "product_support.json").read_text())["manifest"]
    checks = {}
    for split in ("train", "validation", "test"):
        joint = JointRetrievalDataset(product, dest, split)
        joint.validate_site_statistics(dest / "site_target_stats.json")
        indices = [0]
        for i, source in enumerate(joint.sites, 1):
            if len(source):
                indices.append(joint.offsets[i])
        hydro_collate([joint[i] for i in indices])
        if split == "train":
            weights = joint.sampling_weights()
            if not np.isclose(float(weights.sum()), 1):
                raise ValueError("Joint sampling mass does not sum to one")
        checks[split] = dict(
            products=len(joint.product),
            sm=sum(int(s.table.valid_sm.sum()) for s in joint.sites),
            et=sum(int(s.table.valid_et.sum()) for s in joint.sites),
        )
    atomic_json(
        dest / "training_path.json",
        dict(
            status="passed",
            splits=checks,
            scope="joint product/SM/corrected-ET real Zarr batches and training statistics",
        ),
    )
    report["published"] = True
    atomic_json(dest / "split_report.json", report)
    identity = dict(
        version=RELEASE_VERSION,
        et_source=str(root),
        et_collection_sha256=sha256(et_root / "collection.json"),
    )
    atomic_json(dest / "publication.json", identity)
    names = [
        "datasets.json",
        "inventory.parquet",
        "site_target_stats.json",
        "split_report.json",
        "training_path.json",
    ]
    names += [d["manifest"] for d in newdefs]
    marker = dict(
        identity_sha256=fingerprint(identity),
        metadata_sha256={n: sha256(dest / n) for n in names},
        status="published",
        samples=len(inventory),
        sm_unchanged=True,
        et_samples=len(et),
        single_et_zarr=str(et_root / "et.zarr"),
        previous_release=str(old),
        training_path=checks,
    )
    atomic_json(dest / "_PUBLISHED.json", marker)
    return dest, marker


def finalize_and_publish(base, root, scr):
    if (root / "publication_result.json").exists():
        return json.loads((root / "publication_result.json").read_text())
    target = scr / "releases" / root.name
    if active_release(scr).resolve() == target.resolve():
        result = json.loads((target / "_PUBLISHED.json").read_text())
        atomic_json(root / "publication_result.json", result)
        return result
    table = pd.read_parquet(root / "selected.parquet")
    table, locked = assign_et_splits(table, base, root)
    atomic_json(
        root / "final_split.json",
        dict(
            locked_train=sorted(locked),
            splits=table.split.value_counts().to_dict(),
            cross_task_lock=False,
        ),
    )
    output = pack(root, table)
    et_root = scr / "et" / root.name
    et_root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "progress.json", dict(stage="copying_to_scr", samples=len(table)))
    subprocess.run(["rsync", "-rlt", str(output) + "/", str(et_root) + "/"], check=True)
    # No all-file checksum or capacity pass. Real reader checks are part of staging below.
    dest, result = stage_release(base, root, scr, et_root, locked)
    with locked_publication(scr):
        if active_release(scr).resolve() != Path(result["previous_release"]).resolve():
            raise ValueError("Active station release changed while staging")
        atomic_json(
            scr / "ACTIVE.json",
            dict(
                version="site_active_release_v1",
                release=str(dest.relative_to(scr)),
                et_replaced=True,
                sm_unchanged=True,
            ),
        )
    atomic_json(root / "publication_result.json", result)
    atomic_json(
        root / "progress.json", dict(stage="published", samples=len(table), release=str(dest))
    )
    return result


def locked_publication(scr):
    return locked(scr / ".et_replacement.lock", blocking=False)
