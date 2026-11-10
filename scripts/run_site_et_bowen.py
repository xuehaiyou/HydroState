"""Launch the approved ET replacement through collection, acceptance and SCR activation."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from hydrostate.data.gee.contracts import atomic_json, locked  # noqa: E402
from hydrostate.data.site.et_bowen import run, seed_expansion  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", required=True)
p.add_argument("--output", required=True)
p.add_argument("--scr", required=True)
p.add_argument("--resume-from", help="Seed from a published or explicitly stopped run")
p.add_argument("--target-total", type=int, default=3000)
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
with locked(root / ".launcher.lock", blocking=False):
    if a.resume_from:
        seed_expansion(a.resume_from, root, a.scr)
    commit = subprocess.check_output(
        ["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = dict(
        pid=os.getpid(),
        commit=commit,
        started_utc=datetime.now(UTC).isoformat(),
        workers=4,
        cpu_limit=2,
        automatic_publication=True,
        target_total=a.target_total,
        source_code=str(CODE),
        output=str(root),
        scr=a.scr,
    )
    atomic_json(root / "run.json", dict(meta, status="running"))
    try:
        result = run(a.base, root, a.scr, 4, target_total=a.target_total)
        atomic_json(root / "run.json", dict(meta, status="published", result=result))
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(root / "run.json", dict(meta, status="failed", error=repr(error)))
        raise
