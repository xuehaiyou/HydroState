"""Resume the frozen joint draw, then refill failed water-group quotas without re-splitting."""

import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.alignment import grid_metadata, sensor_masks
from hydrostate.data.retrieval.collector import input_arrays
from hydrostate.data.retrieval.contract import PROFILE, RetrievalSchema
from hydrostate.storage import SampleShardWriter

from .archives import sha256
from .candidates import StratifiedPicker
from .contract import VERSION, SiteSchema
from .pilot import site_grid
from .spatial import check_static, footprint_operator, static_weights
from .temporal import epoch


@lru_cache(maxsize=16)
def label_file(path):
    return json.loads(Path(path).read_text())


def load_observations(members):
    """Keep distinct SM sensors; select one ET alternative without averaging source versions."""
    records = []
    ordered = sorted(
        members,
        key=lambda r: (
            not r["operator_ready"],
            {"fluxnet": 0, "icos": 1, "ismn": 2}[r["source"]],
            -(r.get("valid_fraction") or 0),
            r["candidate_id"],
        ),
    )
    et_seen = False
    for member in ordered:
        target = member["target"]
        if target == "et" and et_seen:
            continue
        if target == "sm" and member["source"] != "ismn":
            raise ValueError("Only ISMN SM is authorized")
        label = label_file(member["label_record"])["windows"][member["record_index"]]
        if any(label[k] != member[k] for k in ("target_time", "target_end")):
            raise ValueError("Window/member mismatch")
        if not label[target]["valid"]:
            raise ValueError("Candidate points to an invalid observation")
        if target == "sm" and (
            label["source"] != "ismn" or label["support_type"] not in ("point", "point_layer")
        ):
            raise ValueError("Unexpected SM support")
        records.append(dict(member=member, label=label, target=target))
        et_seen |= target == "et"
    return records


def table_records(table):
    # Preserve binary coordinate precision; JSON decimal round-trips change the native grid.
    return table.astype(object).where(pd.notna(table), None).to_dict("records")


def known_footprint(member, root):
    code = member["site_id"].split(":")[-1]
    path = Path(root) / "stations" / code / "windows.json"
    if not path.exists():
        return None
    return next(
        (
            w
            for w in label_file(str(path))["windows"]
            if w["target_time"] == member["target_time"] and w["target_end"] == member["target_end"]
        ),
        None,
    )


def prediction_support(inputs):
    masks = sensor_masks(inputs)
    # Same input-domain definition as the model and earlier site pilot: any observed slot/sensor.
    observed = np.logical_or.reduce([m.any(axis=0)[0] for m in masks.values()])
    return observed.reshape(40, 3, 40, 3).all(axis=(1, 3))


def prepare_operators(observations, support, spatial, config):
    accepted, rejected = [], []
    for item in observations:
        member, label, target = item["member"], item["label"], item["target"]
        if target == "sm":
            weights = np.zeros((40, 40), dtype="float32")
            weights[20, 20] = 1
            quality = dict(operator="point", accepted=bool(support[20, 20]))
        elif member["operator_ready"]:
            weights, quality = footprint_operator(
                member["footprint_zarr"], support, config["footprint_min_mass"]
            )
        else:
            quality = dict(spatial[member["candidate_id"]])
            weights = static_weights()
            quality["input_support_complete"] = bool(support[weights > 0].all())
            quality["accepted"] &= quality["input_support_complete"]
        provenance = dict(label=label, member=member, spatial_quality=quality)
        if quality["accepted"]:
            accepted.append(
                dict(
                    target=target,
                    value=label[target]["value"],
                    weights=weights,
                    provenance=provenance,
                )
            )
        else:
            rejected.append(provenance)
    return accepted, rejected


def candidate_sample(ee, row, members, root, config, norm):
    root = Path(root)
    cid = row["candidate_id"]
    for folder in ("records", "rejected"):
        path = root / folder / f"{cid}.json"
        if path.exists():
            return json.loads(path.read_text())
    began = monotonic()
    sample = dict(row, **site_grid(row["lon"], row["lat"]))
    if shutil.disk_usage(root).free < config["min_free_gib"] * 1024**3:
        raise OSError("Insufficient free space for station production")
    observations = load_observations(members)
    spatial = {}
    for item in observations:
        m = item["member"]
        if item["target"] == "et" and not m["operator_ready"]:
            known = known_footprint(m, config["footprint_root"])
            if known is not None and not known["footprint_tile_valid"]:
                spatial[m["candidate_id"]] = dict(
                    accepted=False,
                    operator="static_210m",
                    reasons=["known_footprint_failed_no_static_fallback"],
                    footprint=known,
                )
                continue
            spatial[m["candidate_id"]] = check_static(
                ee, sample, item["label"].get("landcover", "unknown"), root / "cache", config
            )

    def reject(reason, **extra):
        result = dict(
            candidate_id=cid,
            status="rejected",
            reason=reason,
            split=row["split"],
            frequency_bin=row["frequency_bin"],
            spatial=spatial,
            elapsed_seconds=monotonic() - began,
            **extra,
        )
        atomic_json(root / "rejected" / f"{cid}.json", result)
        return result

    usable = any(
        i["target"] == "sm"
        or i["member"]["operator_ready"]
        or spatial[i["member"]["candidate_id"]]["accepted"]
        for i in observations
    )
    if not usable:
        return reject("no_accepted_spatial_operator")
    checkpoint = root / "cache" / "inputs" / f"{cid}.npz"
    config_hash = fingerprint(
        {k: v for k, v in config.items() if k not in ("version", "input_profile")}
    )
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["config_hash"]) != config_hash:
                raise ValueError("Input checkpoint configuration mismatch")
            inputs = {k: saved[k] for k in RetrievalSchema().inputs}
            scenes, eligible = json.loads(str(saved["scenes"])), bool(saved["eligible"])
    else:
        inputs, scenes, eligible = input_arrays(ee, sample, root / "cache", norm, config)
        temporary = checkpoint.with_suffix(".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **inputs,
                scenes=json.dumps(scenes),
                eligible=eligible,
                config_hash=config_hash,
            )
        os.replace(temporary, checkpoint)
    if not eligible:
        result = reject("insufficient_input_slot_coverage")
        checkpoint.unlink()
        return result
    accepted, rejected = prepare_operators(
        observations, prediction_support(inputs), spatial, config
    )
    if not accepted:
        result = reject("no_label_with_valid_input_support", rejected_observations=rejected)
        checkpoint.unlink()
        return result
    schema = SiteSchema(observation_capacity=max(Counter(i["target"] for i in accepted).values()))
    targets = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.targets.items()}
    counts = Counter()
    for item in accepted:
        name, index = f"site_{item['target']}", counts[item["target"]]
        targets[f"{name}_native_value"][index] = item["value"]
        targets[f"{name}_native_valid"][index] = 1
        targets[f"{name}_native_weights"][index] = item["weights"]
        counts[item["target"]] += 1
    sid = fingerprint([cid, config_hash])[:24]
    base = root / "zarr" / sid
    # Only this uncommitted candidate owns these paths. Committed records returned above.
    if base.exists():
        shutil.rmtree(base)
    writer = SampleShardWriter(
        str(base / "inputs.zarr"),
        str(base / "labels.zarr"),
        1,
        schema,
        attrs=dict(
            config_sha256=config_hash,
            input_profile=PROFILE,
            input_grids=grid_metadata(sample),
            scenes=scenes,
            observation_records=[i["provenance"] for i in accepted],
            rejected_observations=rejected,
            all_label_alternatives=members,
            normalization_hash=fingerprint(norm),
        ),
    )
    writer.write(inputs, targets)
    writer.finalize()
    result = dict(
        **{
            k: row[k]
            for k in (
                "candidate_id",
                "physical_group_id",
                "spatial_group_id",
                "site_id",
                "split",
                "frequency_bin",
                "landcover",
                "climate_class",
                "lon",
                "lat",
                "target_time",
                "target_end",
                "year",
                "month",
            )
        },
        sampling_group=row["frequency_bin"],
        status="success",
        sample_id=sid,
        version=VERSION,
        input_profile=PROFILE,
        group_id=row["physical_group_id"],
        crs=sample["crs"],
        transform=json.dumps(sample["transform"]),
        input_path=str((base / "inputs.zarr").relative_to(root)),
        label_path=str((base / "labels.zarr").relative_to(root)),
        shard_index=0,
        observation_capacity=schema.observation_capacity,
        valid_sm=counts["sm"],
        valid_et=counts["et"],
        et_support=next(
            (
                i["provenance"]["spatial_quality"]["operator"]
                for i in accepted
                if i["target"] == "et"
            ),
            "none",
        ),
        elapsed_seconds=monotonic() - began,
    )
    atomic_json(root / "records" / f"{cid}.json", result)
    checkpoint.unlink(missing_ok=True)
    return result


def audit_selection(rows, candidates, gap_days):
    table = pd.DataFrame(rows)
    if table.empty:
        return
    if table.candidate_id.duplicated().any() or table.sample_id.duplicated().any():
        raise ValueError("Duplicate production sample")
    expected = candidates.set_index("candidate_id").loc[table.candidate_id]
    for key in ("physical_group_id", "split", "frequency_bin", "lon", "lat", "target_time"):
        if expected[key].tolist() != table[key].tolist():
            raise ValueError(f"Production changed fixed candidate {key}")
    if table.groupby("physical_group_id").split.nunique().gt(1).any():
        raise ValueError("Station crosses splits")
    for _, group in table.groupby("physical_group_id"):
        times = sorted(epoch(v) for v in group.target_time)
        if any(b - a < gap_days * 86400 for a, b in zip(times, times[1:])):
            raise ValueError("Production violates global station date gap")


def refill_round(output, lookup, rows, failures, quotas, config):
    """Freeze each refill draw before QA; failures rebuild capacity only in the next round.

    An admissibility heap cannot reopen dates after a failed test. Planning a full
    maximum-capacity round first keeps that heap monotone and resume order fixed.
    """
    from .sampling_capacity import MaximumDateSchedule

    folder = Path(output) / "refills"
    folder.mkdir(exist_ok=True)
    attempted = {r["candidate_id"] for r in rows + failures}
    rounds = sorted(folder.glob("*.json"))
    if rounds:
        previous = json.loads(rounds[-1].read_text())
        remaining = [cid for cid in previous["candidate_ids"] if cid not in attempted]
        if remaining:
            return remaining
    available = StratifiedPicker(
        list(lookup.values()), seed=config["seed"], gap_days=config["min_date_gap_days"]
    )
    for r in rows + failures:
        available.record(lookup[r["candidate_id"]], success=r["status"] == "success")
    eligible = [
        r
        for r in lookup.values()
        if available.eligible(r)
        and available.counts[r["frequency_bin"]] < quotas.get(r["frequency_bin"], 0)
    ]
    if not eligible:
        return []
    schedule = MaximumDateSchedule(pd.DataFrame(eligible), config["min_date_gap_days"])
    picker = StratifiedPicker(
        list(lookup.values()),
        seed=config["seed"],
        gap_days=config["min_date_gap_days"],
        admissible=schedule.eligible,
    )
    for r in rows + failures:
        picker.record(lookup[r["candidate_id"]], success=r["status"] == "success")
    chosen = []
    while (row := picker.next(quotas)) is not None:
        chosen.append(row["candidate_id"])
        schedule.select(row)
        picker.record(row, success=True)
    atomic_json(
        folder / f"{len(rounds) + 1:04d}.json",
        dict(
            candidate_ids=chosen,
            successes_before=len(rows),
            attempted_before=len(attempted),
            rule="maximum_remaining_date_capacity_then_spatial_year_month_random",
        ),
    )
    return chosen


def collect_production(
    ee, plan, output, config, norm, *, workers=4, max_attempts=None, candidate_ids=None
):
    from .release import lineage

    plan, output = Path(plan), Path(output)
    selected = pd.read_parquet(plan / "selected.parquet").sort_values("draw_order")
    candidates = pd.read_parquet(plan / "input_candidates.parquet")
    members = pd.read_parquet(plan / "label_members.parquet")
    for folder in ("records", "rejected", "cache/inputs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    production = dict(
        config,
        version=VERSION,
        input_profile=PROFILE,
        stage="station_production_v1",
        normalization_hash=fingerprint(norm),
        lineage_sha256=lineage(config, candidates),
        plan_sha256={
            p.name: sha256(p)
            for p in (
                plan / "selected.parquet",
                plan / "input_candidates.parquet",
                plan / "label_members.parquet",
            )
        },
    )
    with locked(output / ".production.lock", blocking=False):
        contract = output / "collection.json"
        if contract.exists() and json.loads(contract.read_text()) != production:
            raise ValueError("Production semantics changed; use a new output directory")
        atomic_json(contract, production)
        rows = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
        failures = [json.loads(p.read_text()) for p in sorted((output / "rejected").glob("*.json"))]
        audit_selection(rows, candidates, config["min_date_gap_days"])
        lookup = {r["candidate_id"]: r for r in table_records(candidates)}
        member_map = defaultdict(list)
        for member in table_records(members):
            member_map[member["input_candidate_id"]].append(member)
        attempted = {r["candidate_id"] for r in rows + failures}
        quotas = selected.frequency_bin.value_counts().to_dict()
        counts = Counter(r["frequency_bin"] for r in rows)
        began, processed = monotonic(), 0

        def snapshot(complete=False):
            if rows:
                temporary = output / "samples.parquet.part"
                pd.DataFrame(rows).to_parquet(temporary, index=False)
                os.replace(temporary, output / "samples.parquet")
            report = dict(
                complete=complete,
                stage=production["stage"],
                successful=len(rows),
                attempted=len(attempted),
                rejected=len(failures),
                quotas=quotas,
                successful_by_bin=dict(counts),
                shortfall={b: max(0, n - counts[b]) for b, n in quotas.items()},
                successful_by_split=dict(Counter(r["split"] for r in rows)),
                et_support=dict(Counter(r["et_support"] for r in rows)),
                valid_sm=sum(r["valid_sm"] for r in rows),
                valid_et=sum(r["valid_et"] for r in rows),
                rejection_reasons=dict(Counter(r["reason"] for r in failures)),
                elapsed_seconds_this_run=monotonic() - began,
                attempts_this_run=processed,
            )
            atomic_json(output / "collection_report.json", report)
            return report

        def consume(result):
            nonlocal processed
            attempted.add(result["candidate_id"])
            processed += 1
            if result["status"] == "success":
                rows.append(result)
                counts[result["frequency_bin"]] += 1
            else:
                failures.append(result)
            atomic_json(
                output / "progress.json",
                dict(
                    attempts=len(attempted),
                    successes=len(rows),
                    attempts_this_run=processed,
                    elapsed_seconds=monotonic() - began,
                    last={k: result[k] for k in ("candidate_id", "status", "frequency_bin")},
                ),
            )
            print(
                json.dumps(
                    dict(
                        attempt=processed,
                        candidate=result["candidate_id"],
                        status=result["status"],
                        successful=len(rows),
                    )
                ),
                flush=True,
            )
            if processed % 100 == 0:
                snapshot()

        def collect(row):
            return candidate_sample(
                ee, row, member_map[row["candidate_id"]], output, production, norm
            )

        initial = [
            lookup[cid]
            for cid in selected.candidate_id
            if cid not in attempted and (candidate_ids is None or cid in candidate_ids)
        ]
        if max_attempts is not None:
            initial = initial[:max_attempts]
        try:
            # Submit only a bounded wave; all original dates are already mutually separated.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for offset in range(0, len(initial), workers):
                    for result in pool.map(collect, initial[offset : offset + workers]):
                        consume(result)
            remaining_original = set(selected.candidate_id) - attempted
            if not remaining_original and candidate_ids is None:
                while True:
                    draw = refill_round(output, lookup, rows, failures, quotas, config)
                    if not draw:
                        break
                    for cid in draw:
                        if max_attempts is not None and processed >= max_attempts:
                            return snapshot()
                        consume(collect(lookup[cid]))
                audit_selection(rows, candidates, config["min_date_gap_days"])
                return snapshot(complete=True)
            return snapshot()
        finally:
            # Crash recovery reads immutable per-candidate commits, including completed workers.
            if not (output / "collection_report.json").exists() or not json.loads(
                (output / "collection_report.json").read_text()
            ).get("complete"):
                snapshot()
