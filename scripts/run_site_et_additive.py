import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
COMMIT = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
SITE = Path("/fossfs/xiaozhen/HydroState/site")
RUN = SITE / "runs/et_additive_20261110"
ROOT = SITE / "prepared/site_et_additive_20261110"
PLAN = SITE / "et_revision_20261110/plan"
sys.path.insert(0, str(CODE))
from hydrostate.data.gee.contracts import atomic_json  # noqa: E402
from hydrostate.data.site.smoke import collect_until_auditable  # noqa: E402

RUN.mkdir(parents=True, exist_ok=True)
with (RUN / ".lock").open("a") as guard:
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = dict(
        pid=os.getpid(),
        commit=COMMIT,
        output=str(ROOT),
        plan=str(PLAN),
        started_utc=datetime.now(UTC).isoformat(),
        workers=4,
        additional_targets=dict(train=828, validation=190, test=182),
        automatic_publication=False,
        cpu_affinity_limit=2,
    )

    def state(status, **extra):
        atomic_json(RUN / "status.json", dict(meta, status=status, **extra))

    try:
        revision = json.loads((PLAN.parent / "revision.json").read_text())
        if revision["additional_targets"] != meta["additional_targets"]:
            raise ValueError("Prepared targets disagree with run configuration")
        # Reuse the cache already prepared by the initial run; no repeated traversal.
        if not (ROOT / "cache/static").is_dir():
            raise FileNotFoundError("Existing additive cache is required")
        env = dict(
            os.environ,
            HYDROSTATE_CODE_ROOT=str(CODE),
            HYDROSTATE_SOURCE_HASHES=str(CODE) + ".sha256",
            HYDROSTATE_PYTHON=sys.executable,
        )
        cmd = [
            "bash",
            str(CODE / "scripts/run_site_production_io.sh"),
            "--plan",
            str(PLAN),
            "--output",
            str(ROOT),
            "--config",
            str(CODE / "configs/data/collection/site_et_additive.py"),
            "--workers",
            "4",
            "--initial-only",
        ]
        if not (RUN / "smoke_audit.json").exists():

            def read_report():
                path = ROOT / "collection_report.json"
                return json.loads(path.read_text()) if path.exists() else {}

            def collect_batch(limit):
                state("smoke_collecting", report=read_report(), total_smoke_budget=240)
                subprocess.run(cmd + ["--max-attempts", str(limit)], env=env, check=True)

            def audit():
                state("smoke_auditing", report=read_report())
                # Keep stderr in the run log so genuine validation errors remain visible.
                with (RUN / "smoke_audit.pending").open("w") as stream:
                    subprocess.run(
                        [
                            sys.executable,
                            str(CODE / "tools/data/prepare_sites.py"),
                            "audit-production",
                            "--root",
                            str(ROOT),
                            "--plan",
                            str(PLAN),
                        ],
                        cwd=CODE,
                        env=dict(env, PYTHONPATH=str(CODE)),
                        check=True,
                        stdout=stream,
                    )
                os.replace(RUN / "smoke_audit.pending", RUN / "smoke_audit.json")

            outcome = collect_until_auditable(read_report, collect_batch, audit)
            if outcome != "passed":
                state(outcome, report=read_report())
                sys.exit(0)
        state("collecting")
        subprocess.run(cmd, env=env, check=True)
        report = json.loads((ROOT / "collection_report.json").read_text())
        state(
            "complete_pending_statistics_and_publication"
            if report["success_targets_met"]
            else "pool_exhausted_with_shortfall",
            report=report,
        )
    except Exception as error:
        state("failed", error=repr(error))
        raise
