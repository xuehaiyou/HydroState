"""Independent corrected-ET production: rare water bins first, reusable inputs second."""

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.contract import PROFILE, RetrievalSchema
from hydrostate.datasets.retrieval_dataset import read_shard_arrays
from hydrostate.storage.reader import ShardCache

from .bowen_support import rematch_footprint
from .candidates import StratifiedPicker
from .contract import VERSION
from .et_candidates import indexed_flux_files
from .pilot import site_grid
from .production import candidate_sample, table_records

ET_VERSION = "site_et_bowen_75_50_spatial90_v1"
KEYS = ["physical_group_id", "target_time", "target_end"]


class PriorityPicker:
    """Land cover precedes climate; lower levels reuse the tested lazy picker."""

    def __init__(self, rows, seed=43):
        self.pickers = {}
        for row in rows:
            self.pickers.setdefault((row["frequency_bin"], row["landcover"]), []).append(row)
        self.pickers = {
            k: StratifiedPicker(v, seed=seed, gap_days=0) for k, v in sorted(self.pickers.items())
        }
        self.counts = Counter()

    def record(self, row, success):
        self.pickers[(row["frequency_bin"], row["landcover"])].record(row, success=success)
        if success:
            self.counts[row["frequency_bin"]] += 1

    def next(self, quotas):
        for b in sorted(quotas, key=lambda k: (self.counts[k] / max(1, quotas[k]), k)):
            if self.counts[b] >= quotas[b]:
                continue
            choices = sorted(
                (k for k in self.pickers if k[0] == b),
                key=lambda k: (self.pickers[k].counts[b], k[1]),
            )
            for key in choices:
                row = self.pickers[key].next({b: quotas[b]})
                if row is not None:
                    return row
        return None


def prepare_plan(base, root, scr):
    plan = root / "plan"
    if (plan / "_PREPARED.json").exists():
        return pd.read_parquet(plan / "candidates.parquet")
    plan.mkdir(parents=True, exist_ok=True)
    audit = base / "reports/bowen_pilot_20261110"
    summary = json.loads((audit / "summary.json").read_text())
    if not summary["complete"] or summary["qc"]["ebr_site_gate"]:
        raise ValueError("Requires completed corrected-label audit without station EBR gate")
    w = pd.read_parquet(audit / "all_windows.parquet")
    w = w[w.corrected_valid & w.frequency_bin.between(0, 6)].copy()
    meta = indexed_flux_files(
        base / "index_v2_20261103",
        base / "splits_20261103",
        base / "strata_20261103",
        ["fluxnet", "icos"],
    )
    fields = [
        "file_id",
        "lon",
        "lat",
        "source_path",
        "source_landcover",
        "utc_offset_hours",
        "spatial_group_id",
        "metadata_file",
        "source_bytes",
        "source_mtime_ns",
    ]
    w = w.merge(meta[fields], on="file_id", validate="many_to_one")
    footprint_path = base / "footprints_icos_20261103/candidates.parquet"
    evidence = pd.read_parquet(footprint_path) if footprint_path.exists() else pd.DataFrame()
    evidence_map = {tuple(r[k] for k in KEYS): r for r in table_records(evidence)}
    w["footprint_preferred"] = [
        r.source == "icos" and (r.physical_group_id, r.target_time, r.target_end) in evidence_map
        for r in w.itertuples()
    ]
    w["_source"] = w.source.map({"fluxnet": 0, "icos": 1})
    w = (
        w.sort_values(
            ["footprint_preferred", "_source", "corrected_measured_fraction", "file_id"],
            ascending=[False, True, False, True],
        )
        .drop_duplicates(KEYS)
        .copy()
    )
    w["candidate_id"] = [
        fingerprint([r.physical_group_id, r.lon, r.lat, r.target_time, r.target_end])[:24]
        for r in w.itertuples()
    ]
    w["split"] = "train"  # Final group-stratified selection is made after input acceptance.
    w["year"] = pd.to_datetime(w.target_time).dt.year
    w["month"] = pd.to_datetime(w.target_time).dt.month
    w["has_sm"], w["has_et"] = False, True
    for column in ["frequency_bin", "landcover", "climate_class"]:
        w[column] = w[column].fillna(-1).astype(int)
    w.to_parquet(plan / "candidates.parquet", index=False)
    atomic_json(plan / "footprints.json", {"rows": list(evidence_map.values())})
    # Snapshot successful input manifests once, including the running old supplement.
    reused = []
    for name in ["site_5x8_am_v1", "site_5x8_am_supplement_20261108", "site_et_additive_20261110"]:
        source = base / "prepared" / name
        if not (source / "samples.parquet").exists():
            continue
        table = pd.read_parquet(source / "samples.parquet")
        table["reuse_root"] = str(source)
        reused.append(table)
    pd.concat(reused, ignore_index=True).drop_duplicates("candidate_id", keep="last").to_parquet(
        plan / "reusable.parquet", index=False
    )
    old = pd.read_parquet(scr / "inventory.parquet")
    previous_train = set(old.loc[old.valid_et.gt(0) & old.split.eq("train"), "physical_group_id"])
    prior = base / "et_revision_20261110/inventory.parquet"
    if prior.exists():
        old = pd.read_parquet(prior)
        previous_train.update(
            old.loc[old.valid_et.gt(0) & old.split.eq("train"), "physical_group_id"]
        )
    atomic_json(plan / "previous_et_training.json", sorted(previous_train))
    atomic_json(plan / "_PREPARED.json", dict(candidates=len(w), label_version=ET_VERSION))
    return w


class InputReuse:
    def __init__(self, path, norm_hash):
        self.lookup = {r["candidate_id"]: r for r in table_records(pd.read_parquet(path))}
        self.norm_hash = norm_hash
        self.local = threading.local()
        self.configs = {}
        for root in {r["reuse_root"] for r in self.lookup.values()}:
            config = json.loads((Path(root) / "collection.json").read_text())
            self.configs[root] = fingerprint(
                {k: v for k, v in config.items() if k not in ("version", "input_profile")}
            )

    def __call__(self, row):
        old = self.lookup.get(row["candidate_id"])
        if old is None:
            return None
        if old["input_profile"] != PROFILE:
            raise ValueError("Input reuse requires the exact current input profile")
        source = Path(old["reuse_root"])
        group = zarr.open_group(str(source / old["input_path"]), mode="r")
        if group.attrs["normalization_hash"] != self.norm_hash:
            raise ValueError("Input normalization changed; cannot reuse normalized inputs")
        if not hasattr(self.local, "cache"):
            self.local.cache = ShardCache()
        inputs = read_shard_arrays(
            source,
            self.local.cache,
            pd.Series(old),
            "input",
            RetrievalSchema().inputs,
            VERSION,
            self.configs[str(source)],
        )
        grid = site_grid(row["lon"], row["lat"])
        if old["crs"] != grid["crs"] or not np.allclose(
            json.loads(old["transform"]), grid["transform"], rtol=0, atol=1e-7
        ):
            raise ValueError("Reused inputs are not on the same native grid")
        return inputs, dict(group.attrs["scenes"]), True


def observation(row, evidence_map, root):
    meta = dict(row)
    label = dict(
        source=row["source"],
        site_id=row["site_id"],
        group_id=row["physical_group_id"],
        lon=row["lon"],
        lat=row["lat"],
        landcover=row["source_landcover"],
        target_time=row["target_time"],
        target_end=row["target_end"],
        utc_offset_hours=row["utc_offset_hours"],
        time_basis="local_standard",
        source_file=dict(path=row["source_path"]),
        correction="provider_LE_CORR_ONEFlux",
        label_version=ET_VERSION,
    )
    et = dict(
        valid=True,
        value=row["corrected_value"],
        **{
            k: row["corrected_" + k]
            for k in ["valid_fraction", "measured_fraction", "day_fraction", "night_fraction"]
        },
    )
    label.update(
        et=et,
        et_corrected=dict(et),
        et_uncorrected=dict(
            value=row["raw_matched_value"], valid=True, time_support="same_as_corrected"
        ),
    )
    member = {k: row[k] for k in ["candidate_id", "site_id", "source", "target_time", "target_end"]}
    member.update(target="et", operator_ready=False)
    evidence = evidence_map.get(tuple(row[k] for k in KEYS)) if row["footprint_preferred"] else None
    if evidence is not None:
        if abs(evidence["lon"] - row["lon"]) > 1e-7 or abs(evidence["lat"] - row["lat"]) > 1e-7:
            return None, dict(reason="footprint_coordinate_mismatch")
        support, rejection = rematch_footprint(row, meta, evidence, root)
        if rejection:
            return None, rejection
        member.update(operator_ready=True, footprint_zarr=support["path"])
        label.update(
            et=support["et"], et_corrected=support["et"], et_uncorrected=support["et_uncorrected"]
        )
    return dict(member=member, label=label, target="et"), None


def enrich_record(result, root):
    """Recover provenance columns even if interrupted just after the sample commit."""
    if result.get("label_version") == ET_VERSION:
        return result
    if result["status"] == "success":
        attrs = zarr.open_group(str(root / result["label_path"]), mode="r").attrs
        evidence = attrs["observation_records"][0]
        q = evidence["spatial_quality"]
        label = evidence["label"]
        result.update(
            homogeneous_fraction=q.get("homogeneous_fraction"),
            classification_valid_fraction=q.get("valid_fraction"),
            dominant_landcover=q.get("dominant_class"),
            et_corrected=label["et"]["value"],
            et_uncorrected=label["et_uncorrected"]["value"],
            et_valid_fraction=label["et"]["valid_fraction"],
            et_measured_fraction=label["et"]["measured_fraction"],
            et_day_fraction=label["et"]["day_fraction"],
            et_night_fraction=label["et"]["night_fraction"],
            label_version=ET_VERSION,
        )
        atomic_json(root / "records" / f"{result['candidate_id']}.json", result)
    return result


def collect(ee, table, root, config, norm, workers=4):
    evidence = json.loads((root / "plan/footprints.json").read_text())["rows"]
    evidence_map = {tuple(r[k] for k in KEYS): r for r in evidence}
    rows = table_records(table)
    lookup = {r["candidate_id"]: r for r in rows}
    picker = PriorityPicker(rows, seed=config["seed"])
    ledger = root / "attempts.jsonl"
    completed = []
    recorded = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            event = json.loads(line)
            cid = event["candidate_id"]
            picker.record(lookup[cid], event["status"] == "success")
            recorded.add(cid)
            if event["status"] == "success":
                completed.append(
                    enrich_record(json.loads((root / "records" / f"{cid}.json").read_text()), root)
                )
    # Recover a batch interrupted after a successful atomic sample commit.
    for folder in ["records", "rejected"]:
        for path in sorted((root / folder).glob("*.json")):
            if path.stem not in recorded:
                event = json.loads(path.read_text())
                success = event["status"] == "success"
                picker.record(lookup[path.stem], success)
                with ledger.open("a") as f:
                    f.write(json.dumps(dict(candidate_id=path.stem, status=event["status"])) + "\n")
                recorded.add(path.stem)
                if success:
                    completed.append(enrich_record(event, root))
    reuse = InputReuse(root / "plan/reusable.parquet", fingerprint(norm))

    def work(row):
        item, rejection = observation(row, evidence_map, root)
        if rejection:
            result = dict(candidate_id=row["candidate_id"], status="rejected", **rejection)
            atomic_json(root / "rejected" / f"{row['candidate_id']}.json", result)
            return result
        result = candidate_sample(
            ee, row, [item["member"]], root, config, norm, observations=[item], reuse_inputs=reuse
        )
        return enrich_record(result, root)

    def save(stage, quotas):
        if completed:
            pd.DataFrame(completed).to_parquet(root / "samples.parquet", index=False)
            if not (root / "smoke.json").exists():
                from hydrostate.datasets.site_dataset import SiteDataset

                dataset = SiteDataset(root / "samples.parquet", "train")
                item = dataset[0]
                if item["data_samples"]["site_et_native_valid"].sum() != 1:
                    raise ValueError("Corrected-ET smoke failed actual dataset read")
                atomic_json(
                    root / "smoke.json",
                    dict(
                        status="passed",
                        sample_id=completed[0]["sample_id"],
                        scope="actual corrected ET Zarr input, masks and operator read",
                    ),
                )
        atomic_json(
            root / "progress.json",
            dict(
                stage=stage,
                attempted=len(recorded),
                successful=len(completed),
                successful_by_bin=dict(picker.counts),
                quotas=quotas,
            ),
        )

    nonzero = {int(b): int(n) for b, n in table.frequency_bin.value_counts().items() if b > 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stage in ["nonzero", "zero"]:
            # Do not let a group with only a few windows cap all other groups.
            quotas = (
                nonzero
                if stage == "nonzero"
                else {0: max((picker.counts[b] for b in nonzero), default=0)}
            )
            if stage == "zero" and quotas[0] == 0:
                raise ValueError(
                    "No usable nonzero-frequency ET; cannot silently publish a zero-only draw"
                )
            save(stage, quotas)
            while True:
                batch = []
                for _ in range(workers):
                    row = picker.next(quotas)
                    if row is None:
                        break
                    # Reserve without counting a success; deterministic completion order below.
                    picker.record(row, False)
                    batch.append(row)
                if not batch:
                    break
                for row, result in zip(batch, pool.map(work, batch), strict=True):
                    success = result["status"] == "success"
                    if success:
                        picker.record(row, True)
                        completed.append(result)
                    recorded.add(row["candidate_id"])
                    with ledger.open("a") as f:
                        f.write(
                            json.dumps(
                                dict(candidate_id=row["candidate_id"], status=result["status"])
                            )
                            + "\n"
                        )
                save(stage, quotas)
                print(
                    json.dumps(dict(stage=stage, attempted=len(recorded), success=len(completed))),
                    flush=True,
                )
    # A final parallel batch can exceed a quota by at most workers-1; retain cached surplus only.
    result = pd.DataFrame(completed)
    zero_limit = max(picker.counts[b] for b in nonzero)
    selected = result[
        ~result.frequency_bin.eq(0) | (result.frequency_bin.eq(0).cumsum() <= zero_limit)
    ]
    selected.to_parquet(root / "selected.parquet", index=False)
    atomic_json(
        root / "collection_report.json",
        dict(
            complete=True,
            successful=len(selected),
            cached_surplus=len(result) - len(selected),
            by_bin=selected.frequency_bin.value_counts().to_dict(),
            zero_target=zero_limit,
            attempted=len(recorded),
            minimum_date_gap_days=0,
        ),
    )
    return selected


def run(base, root, scr, workers):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    from hydrostate.data.gee.client import initialize

    from .et_bowen_release import finalize_and_publish

    base, root, scr = map(Path, (base, root, scr))
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".run.lock", blocking=False):
        table = prepare_plan(base, root, scr)
        norm = load_computed_config()
        config = dict(
            version=VERSION,
            input_profile=PROFILE,
            label_version=ET_VERSION,
            seed=20261110,
            min_date_gap_days=0,
            homogeneous_fraction=0.90,
            spatial_valid_fraction=0.90,
            class_confidence=0.50,
            footprint_min_mass=0.95,
            min_free_gib=0,
            min_input_fraction=0.50,
            min_active_slots=2,
            footprint_root=str(base / "footprints_icos_20261103"),
            normalization_hash=fingerprint(norm),
            reuse_cache_roots=[
                str(base / "prepared" / name / "cache")
                for name in [
                    "site_et_additive_20261110",
                    "site_5x8_am_supplement_20261108",
                    "site_5x8_am_v1",
                ]
            ],
            temporal_qc=dict(local_day=0.75, day=0.75, night=0.75, measured=0.50),
            split_policy="task_specific_spatial_groups_no_cross_task_lock",
            sampling_priority=["frequency", "landcover", "climate", "year", "month", "site"],
        )
        path = root / "collection.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Production semantics changed; use a new version directory")
        atomic_json(path, config)
        for folder in ["records", "rejected", "cache/inputs"]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        if not (root / "collection_report.json").exists():
            collect(initialize("cropland-477906"), table, root, config, norm, workers)
        return finalize_and_publish(base, root, scr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scr", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(run(a.base, a.output, a.scr, a.workers)), flush=True)


if __name__ == "__main__":
    main()
