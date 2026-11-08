import json
from pathlib import Path

import pandas as pd
import pytest
import zarr
from test_joint_training import joint as joint

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.site import publication
from hydrostate.data.site.archives import sha256
from hydrostate.data.site.statistics import cohort_statistics


def expanded(table):
    frames = []
    for split in ("train", "validation", "test"):
        frame = table.copy()
        frame["split"] = split
        for key in ("sample_id", "candidate_id", "spatial_group_id", "physical_group_id"):
            if key in frame:
                frame[key] = frame[key].astype(str) + "_" + split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def bundle(joint, monkeypatch, tmp_path):
    dataset, _, report = joint
    root, split = dataset.sites[0].root, dataset.site_root
    product = Path(dataset.manifest)
    ptable = expanded(pd.read_parquet(product))
    ptable.to_parquet(product, index=False)
    proof = Path(report["joint_product_selection"]).parent
    ptable[["sample_id", "split"]].to_parquet(proof / "selection.parquet", index=False)
    atomic_json(proof / "product_support.json", dict(manifest_sha256=sha256(product)))
    table = expanded(pd.read_parquet(root / "samples.parquet"))
    table.to_parquet(root / "samples.parquet", index=False)
    table.to_parquet(split / "cohort_0.parquet", index=False)
    table.to_parquet(split / "inventory.parquet", index=False)
    table.to_parquet(split / "station_splits.parquet", index=False)
    table.assign(product_overlap_count=0, split_reason="seeded").to_parquet(
        proof / "station_splits.parquet", index=False
    )
    plan = tmp_path / "plan"
    plan.mkdir()
    for name in ("selected.parquet", "input_candidates.parquet", "label_members.parquet"):
        table.to_parquet(plan / name, index=False)
    config = publication.read(root / "collection.json")
    config.update(
        product_manifest=str(product),
        plan_sha256={p.name: sha256(p) for p in plan.glob("*.parquet")},
    )
    atomic_json(root / "collection.json", config)
    config_hash = fingerprint(
        {k: v for k, v in config.items() if k not in ("version", "input_profile")}
    )
    for field in ("input_path", "label_path"):
        for name in table[field].unique():
            zarr.open_group(str(root / name), mode="a").attrs["config_sha256"] = config_hash
    for folder in ("records", "rejected"):
        (root / folder).mkdir()
    for row in table.itertuples():
        atomic_json(root / "records" / f"{row.candidate_id}.json", dict(sample_id=row.sample_id))
    atomic_json(
        root / "collection_report.json",
        dict(initial_screening_complete=True, complete=False, successful=6, shortfall={}),
    )
    atomic_json(root / "acceptance.json", dict(status="passed", samples=2))  # stale pilot
    report.update(
        samples=6,
        source_manifests={str(root): sha256(root / "samples.parquet")},
        joint_product_selection_sha256=sha256(proof / "selection.parquet"),
        source_splits_sha256=sha256(proof / "station_splits.parquet"),
    )
    atomic_json(split / "split_report.json", report)
    stats = cohort_statistics(publication.read(split / "datasets.json")["datasets"])
    atomic_json(split / "site_target_stats.json", stats)
    preflight = tmp_path / "preflight.json"
    atomic_json(
        preflight,
        dict(
            status="passed",
            scope="actual_mixed_Zarr_batches",
            site_artifacts={
                n: sha256(split / n)
                for n in ("split_report.json", "site_target_stats.json", "inventory.parquet")
            },
            statistics=dict(site=stats),
        ),
    )
    audits = []

    def audit(source, candidate_plan, *, require_screened):
        assert require_screened and source == root and candidate_plan == plan
        audits.append(source)
        receipt = dict(
            audit_version="site_full_array_audit_v2",
            status="passed",
            screening_checked=True,
            samples=6,
            manifest_sha256=sha256(source / "samples.parquet"),
            collection_sha256=fingerprint(publication.read(source / "collection.json")),
            collection_report_sha256=sha256(source / "collection_report.json"),
        )
        atomic_json(source / "acceptance.json", receipt)
        return receipt

    monkeypatch.setattr(publication, "audit_production", audit)
    return (
        dict(
            split_root=split,
            plans=[plan],
            output=tmp_path / "published",
            preflight=preflight,
            code_commit="a" * 40,
            reserve_gib=0,
        ),
        root,
        audits,
    )


def test_dry_run_stale_audit_and_no_source_mutation(bundle):
    args, root, audits = bundle
    before = sha256(args["split_root"] / "site_target_stats.json")
    check = publication.publish_joint(**args, check_only=True)
    assert check["pending_full_audits"] == [str(root)]
    assert not args["output"].exists() and audits == []
    result = publication.publish_joint(**args)
    assert result["samples"] == 6 and result["by_split"] == dict(train=2, validation=2, test=2)
    assert audits == [root]
    assert sha256(args["split_root"] / "site_target_stats.json") == before
    assert sha256(args["output"] / "site_target_stats.json") == before
    assert (
        publication.read(args["output"] / "datasets.json")["datasets"][0]["data_root"]
        == "cohorts/0"
    )
    publication.require_publication_ready(args["output"])
    assert publication.publish_joint(**args) == result
    assert audits == [root]
    (args["output"] / "datasets.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        publication.require_publication_ready(args["output"])


def test_interrupted_copy_is_not_published_and_resumes_with_checksum_repair(bundle, monkeypatch):
    args, root, audits = bundle
    original = publication.copy_verified
    calls = []

    def interrupted(source, target, names):
        original(source, target, names)
        calls.append(target)
        if len(calls) == 1:
            raise OSError("Simulated interruption after cohort copy")

    monkeypatch.setattr(publication, "copy_verified", interrupted)
    with pytest.raises(OSError, match="interruption"):
        publication.publish_joint(**args)
    assert not (args["output"] / "_PUBLISHED.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        publication.require_publication_ready(args["output"])
    copied = args["output"] / "cohorts/0/collection.json"
    copied.write_text("corrupt partial file")
    monkeypatch.setattr(publication, "copy_verified", original)
    publication.publish_joint(**args)
    assert sha256(copied) == sha256(root / "collection.json")
    assert audits == [root]  # Full receipt is reused, not the stale pilot receipt.


def test_mismatched_preflight_and_low_capacity_never_publish(bundle, monkeypatch):
    args, _, audits = bundle
    original = args["preflight"].read_text()
    invalid = json.loads(original)
    invalid["site_artifacts"]["inventory.parquet"] = "0" * 64
    atomic_json(args["preflight"], invalid)
    with pytest.raises(ValueError, match="preflight does not match"):
        publication.publish_joint(**args)
    args["preflight"].write_text(original)
    monkeypatch.setattr(publication, "capacity", lambda *a: dict(enough_space=False))
    with pytest.raises(OSError, match="capacity"):
        publication.publish_joint(**args)
    assert not args["output"].exists() and audits == []


def test_two_cohorts_keep_distinct_collection_hashes_and_v3_statistics(bundle, monkeypatch):
    import shutil

    args, first, _ = bundle
    split = args["split_root"]
    second = first.parent / "site_second"
    shutil.copytree(first, second)
    table = pd.read_parquet(second / "samples.parquet")
    for key in ("sample_id", "candidate_id", "physical_group_id", "spatial_group_id"):
        table[key] += "_second"
    table["cohort"] = 1
    table.to_parquet(second / "samples.parquet", index=False)
    table.to_parquet(split / "cohort_1.parquet", index=False)
    old = pd.read_parquet(split / "inventory.parquet")
    pd.concat([old, table], ignore_index=True).to_parquet(split / "inventory.parquet", index=False)
    plan = first.parent / "plan_second"
    plan.mkdir()
    for name in ("selected.parquet", "input_candidates.parquet", "label_members.parquet"):
        table.to_parquet(plan / name, index=False)
    config = publication.read(second / "collection.json")
    config["class_confidence"] = 0.5
    config["plan_sha256"] = {p.name: sha256(p) for p in plan.glob("*.parquet")}
    atomic_json(second / "collection.json", config)
    digest = fingerprint({k: v for k, v in config.items() if k not in ("version", "input_profile")})
    for field in ("input_path", "label_path"):
        for name in table[field].unique():
            zarr.open_group(str(second / name), mode="a").attrs["config_sha256"] = digest
    shutil.rmtree(second / "records")
    (second / "records").mkdir()
    for row in table.itertuples():
        atomic_json(second / "records" / f"{row.candidate_id}.json", dict(sample_id=row.sample_id))
    report = publication.read(split / "split_report.json")
    report["source_manifests"][str(second)] = sha256(second / "samples.parquet")
    report["samples"] = 12
    atomic_json(split / "split_report.json", report)
    definitions = publication.read(split / "datasets.json")["datasets"]
    definitions.append(
        dict(type="SiteDataset", manifest=str(split / "cohort_1.parquet"), data_root=str(second))
    )
    atomic_json(split / "datasets.json", dict(datasets=definitions))
    atomic_json(split / "site_target_stats.json", cohort_statistics(definitions))
    check = publication.read(args["preflight"])
    check["site_artifacts"] = {n: sha256(split / n) for n in check["site_artifacts"]}
    check["statistics"]["site"] = publication.read(split / "site_target_stats.json")
    atomic_json(args["preflight"], check)
    args["plans"].append(plan)
    audited = []

    def audit(root, plan, *, require_screened):
        audited.append(root)
        atomic_json(
            root / "acceptance.json",
            dict(
                audit_version="site_full_array_audit_v2",
                status="passed",
                screening_checked=True,
                samples=6,
                manifest_sha256=sha256(root / "samples.parquet"),
                collection_sha256=fingerprint(publication.read(root / "collection.json")),
                collection_report_sha256=sha256(root / "collection_report.json"),
            ),
        )

    monkeypatch.setattr(publication, "audit_production", audit)
    result = publication.publish_joint(**args)
    assert result["samples"] == 12 and audited == [first, second]
    assert result["by_split"] == dict(train=4, validation=4, test=4)
    for i, root in enumerate([first, second]):
        assert sha256(args["output"] / f"cohorts/{i}/collection.json") == sha256(
            root / "collection.json"
        )
    assert sha256(args["output"] / "site_target_stats.json") == sha256(
        split / "site_target_stats.json"
    )


def test_explicit_capacity_bypass_never_scans_sizes_and_still_audits(bundle, monkeypatch):
    args, root, audits = bundle

    def forbidden(*args):
        raise AssertionError("Capacity scan must not run")

    monkeypatch.setattr(publication, "capacity", forbidden)
    result = publication.publish_joint(**args, skip_capacity_check=True)
    assert result["space"]["status"] == "skipped_by_user"
    assert result["status"] == "passed" and audits == [root]
