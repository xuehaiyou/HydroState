"""Bounded sequential multi-station development collection with successful-bin quotas."""

import json
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .candidates import StratifiedPicker
from .pilot import collect_pilot


def collect_stations(ee, candidates, output, config, norm, *, per_bin=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(candidates)
    table = table[table.split.eq("train")]
    plan = dict(
        candidate_sha256=sha256(candidates),
        per_bin=per_bin,
        seed=config["seed"],
        split="train",
        stage="development",
        date_gap_days=config["min_date_gap_days"],
    )
    with locked(output / ".sampling.lock", blocking=False):
        dest = output / "sampling_plan.json"
        if dest.exists() and json.loads(dest.read_text()) != plan:
            raise ValueError("Pilot sampling plan changed; use a new output directory")
        atomic_json(dest, plan)
        ledger = output / "sampling_attempts.json"
        attempts = json.loads(ledger.read_text()) if ledger.exists() else []
        lookup = {r["candidate_id"]: r for r in table.to_dict("records")}
        picker = StratifiedPicker(
            list(lookup.values()), seed=config["seed"], gap_days=config["min_date_gap_days"]
        )
        # Recover commits even if a process ended before recording the sampling attempt.
        completed = {
            json.loads(p.read_text())["candidate_id"] for p in (output / "records").glob("*.json")
        }
        seen = set()
        for item in attempts:
            key = item["candidate_id"]
            picker.record(lookup[key], success=key in completed)
            seen.add(key)
        for key in sorted(completed - seen):
            picker.record(lookup[key], success=True)
            attempts.append(dict(candidate_id=key, status="success_recovered"))
        quotas = {b: per_bin for b in range(7)}
        production = dict(
            config,
            candidate_manifest_sha256=plan["candidate_sha256"],
            sampling_plan_sha256=fingerprint(plan),
        )
        while row := picker.next(quotas):
            record = json.loads(Path(row["label_record"]).read_text())["windows"][
                row["record_index"]
            ]
            record = dict(
                record,
                group_id=row["physical_group_id"],
                split=row["split"],
                candidate_id=row["candidate_id"],
                spatial_group_id=row["spatial_group_id"],
            )
            accepted = collect_pilot(ee, [record], output, production, norm, allow_empty=True)
            success = any(r.get("candidate_id") == row["candidate_id"] for r in accepted)
            picker.record(row, success=success)
            attempts.append(
                dict(
                    candidate_id=row["candidate_id"],
                    status="success" if success else "input_rejected",
                )
            )
            atomic_json(ledger, attempts)
            atomic_json(
                output / "collection_report.json",
                dict(
                    complete=False,
                    quotas=quotas,
                    successful=dict(picker.counts),
                    attempts=len(attempts),
                ),
            )
        report = dict(
            complete=True,
            stage="pilot_unreleased",
            quotas=quotas,
            successful={b: picker.counts[b] for b in quotas},
            shortfall={b: max(0, quotas[b] - picker.counts[b]) for b in quotas},
            attempts=len(attempts),
            reason="Missing groups remain explicit; no frequency reassignment or QC relaxation",
        )
        atomic_json(output / "collection_report.json", report)
        return report
