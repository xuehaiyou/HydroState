"""Relax an existing region's observation age limit without repeating scene selection."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sqlite3
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

PENDING = "recency_update_pending.json"
# Recency is applied after scene selection, so the stored ages and slots suffice.
ACCEPTED = """COALESCE(json_extract(result, '$.active_slots') >= ?
    AND json_extract(result, '$.latest_age_days') IS NOT NULL
    AND (? IS NULL OR json_extract(result, '$.latest_age_days') <= ?), 0)"""


def _apply(root, record):
    plan = record["new_plan"]
    cfg = plan["config"]
    with sqlite3.connect(root / "preparation.sqlite") as db:
        db.execute(
            f"""UPDATE windows SET result=json_set(result, '$.accepted',
                json(CASE WHEN {ACCEPTED} THEN 'true' ELSE 'false' END))
                WHERE json_extract(result, '$.accepted') != {ACCEPTED}""",
            (cfg["min_active_slots"], cfg["latest_days"], cfg["latest_days"]) * 2,
        )
    # Newly accepted windows need complete bands before a ready manifest can exist.
    # Originals are in the revision backup; repeated invalidation is safe on resume.
    for relative in record["invalidated_files"]:
        (root / relative).unlink(missing_ok=True)
    atomic_json(root / "plan.json", plan)
    atomic_json(root / "selection_contract.json", record["new_contract"])
    record["completed_at"] = datetime.now(UTC).isoformat()
    atomic_json(Path(record["backup"]) / "revision.json", record)
    (root / PENDING).unlink()
    return record


def update_recency(root, latest_days):
    """Offline, locked metadata revision; rerun the same command after interruption.

    Only relaxation is supported. Existing prediction products require a separate
    output revision; this tool never relabels already computed predictions.
    """
    if latest_days is not None and (
        isinstance(latest_days, bool) or not math.isfinite(latest_days) or latest_days < 0
    ):
        raise ValueError("latest_days must be null or finite and nonnegative")
    root = Path(root).resolve()
    if not (root / "plan.json").is_file():
        raise ValueError("An existing regional plan is required")
    with ExitStack() as locks:
        for name in ("prepare.lock", "predict.lock", "download.lock"):
            locks.enter_context(locked(root / name, blocking=False))
        pending = root / PENDING
        if pending.exists():
            record = json.loads(pending.read_text())
            if record["new_plan"]["config"]["latest_days"] != latest_days:
                raise ValueError("Resume the pending recency revision with its original limit")
            return _apply(root, record)
        plan = json.loads((root / "plan.json").read_text())
        if plan["config"].get("temporal_sampling", "past_16d_v1") != "past_16d_v1":
            raise ValueError("Recency revision only supports past_16d_v1; use a separate plan")
        old = plan["config"]["latest_days"]
        if old == latest_days:
            return dict(changed=False, latest_days=latest_days)
        if old is None or (latest_days is not None and latest_days < old):
            raise ValueError("This operation only relaxes the existing age limit")
        if any((root / "products").glob("*.zarr")):
            raise ValueError("Existing predictions require a separate output revision")
        contract = json.loads((root / "selection_contract.json").read_text())
        if contract["plan"] != fingerprint(plan) or contract["selection_version"] != 1:
            raise ValueError("Existing selection contract does not match the plan")
        if plan["identity"]["config"] != plan["config"]:
            raise ValueError("Plan identity/config mismatch")
        new_plan = copy.deepcopy(plan)
        new_plan["config"]["latest_days"] = latest_days
        new_plan["identity"]["config"]["latest_days"] = latest_days
        del new_plan["plan_hash"]
        new_plan["plan_hash"] = fingerprint(new_plan)
        new_contract = dict(contract, plan=fingerprint(new_plan))
        params = (plan["config"]["min_active_slots"], latest_days, latest_days)
        with sqlite3.connect(root / "preparation.sqlite") as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Preparation database failed its integrity check")
            mismatch = db.execute(
                f"SELECT count(*) FROM windows WHERE json_extract(result, '$.accepted') "
                f"!= {ACCEPTED}",
                (params[0], old, old),
            ).fetchone()[0]
            if mismatch:
                raise ValueError("Stored acceptance differs from the original selection rule")
            dates = [
                dict(date=day, windows=n, accepted_before=before, accepted_after=after)
                for day, n, before, after in db.execute(
                    f"SELECT date, count(*), sum(json_extract(result, '$.accepted')), "
                    f"sum({ACCEPTED}) FROM windows GROUP BY date ORDER BY date",
                    params,
                )
            ]
            affected = {r["date"] for r in dates if r["accepted_before"] != r["accepted_after"]}
            backup = root / "revisions" / datetime.now(UTC).strftime("recency_%Y%m%dT%H%M%S%fZ")
            backup.mkdir(parents=True)
            with sqlite3.connect(backup / "preparation.sqlite") as snapshot:
                db.backup(snapshot)
        invalidated = []
        for day in sorted(affected):
            for relative in (
                f"manifests/{day}.parquet",
                f"selections/{day}.parquet",
                f"reports/preparation_{day}.json",
            ):
                if (root / relative).exists():
                    invalidated.append(relative)
        for relative in ["plan.json", "selection_contract.json", *invalidated]:
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
        record = dict(
            changed=True,
            old_latest_days=old,
            backup=str(backup),
            started_at=datetime.now(UTC).isoformat(),
            new_plan=new_plan,
            new_contract=new_contract,
            dates=dates,
            invalidated_files=invalidated,
        )
        atomic_json(pending, record)
        return _apply(root, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    rule = parser.add_mutually_exclusive_group(required=True)
    rule.add_argument("--latest-days", type=float)
    rule.add_argument("--no-recency-limit", action="store_true",
                      help="Use slot/QA/active-slot rules only, preserving observed ages")
    args = parser.parse_args()
    report = update_recency(args.root, args.latest_days)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"new_plan", "new_contract"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
