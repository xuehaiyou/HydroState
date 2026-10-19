#!/usr/bin/env python
"""Submit the frozen experiment plan as GPU account capacity becomes available."""

import argparse
import fcntl
import getpass
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tools.submit_shared_private import save_record, submit_stage
from tools.submit_sharing_ablation import summary_command, training_command


def submission_limits():
    output = subprocess.check_output([
        "sacctmgr", "-n", "-P", "show", "qos", "gpu",
        "format=MaxSubmitJobsPA,MaxSubmitJobsPU"], text=True).strip()
    account, user = output.split("|")[:2]
    return dict(account=int(account) if account else None, user=int(user) if user else None,
                username=getpass.getuser(), checked_utc=datetime.now(UTC).isoformat())


def capacity(record):
    resources = record["gpu_resources"]
    output = subprocess.check_output([
        "squeue", "--array", "--noheader", f"--account={resources['account']}",
        f"--qos={resources['qos']}", "--format=%i|%u"], text=True)
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    limits = record["gpu_submission_limits"]
    owned = sum(user == limits["username"] for _, user in rows)
    free = [len(record["experiments"])]
    if limits["account"] is not None:
        free.append(limits["account"] - len(rows))
    if limits["user"] is not None:
        free.append(limits["user"] - owned)
    return dict(free=max(0, min(free)), account_jobs=len(rows), user_jobs=owned,
                job_ids=sorted(job for job, _ in rows), checked_utc=datetime.now(UTC).isoformat())


def quota_rejection(entry):
    return (entry.get("returncode", 0) != 0 and not entry.get("job_id")
            and any(reason in entry.get("stderr", "") for reason in
                    ("MaxSubmitJobsPerAccount", "QOSMaxSubmitJobPerUserLimit")))


def dispatch_pending(path, environment=None):
    # One writer per run, including interactive recovery and scheduler callbacks.
    with (path.parent / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(path.read_text())
        # A queued controller already owns future submission. Only that controller
        # may continue; invoking this tool twice cannot create a second chain.
        controller = record.get("active_dispatcher")
        if controller and os.environ.get("SLURM_JOB_ID") != controller:
            print(f"Existing quota controller: {controller}; inspect Slurm before recovery")
            return
        record.pop("active_dispatcher", None)
        environment = dict(os.environ) if environment is None else dict(environment)
        environment.update(record["environment"])
        for entry in record["experiments"]:
            stage = entry["stage"]
            previous = record["jobs"].get(stage)
            if previous and previous.get("job_id"):
                continue
            if previous and not quota_rejection(previous):
                raise RuntimeError(f"Inspect ambiguous/non-quota submission before retry: {stage}")
            snapshot = capacity(record)
            record["last_capacity_check"] = snapshot
            save_record(path, record)
            if (snapshot["free"] == 0 or snapshot["job_ids"] ==
                    record.get("last_quota_rejection_job_ids")):
                break
            if previous:
                history = record.setdefault("past_rejected_attempts", {}).setdefault(stage, [])
                history.append(previous)
                del record["jobs"][stage]
                save_record(path, record)
            try:
                submit_stage(record, path, stage, training_command(record, entry), environment)
            except subprocess.CalledProcessError:
                if not quota_rejection(record["jobs"][stage]):
                    raise
                # A different user may have taken the last slot after our query.
                record["last_quota_rejection_job_ids"] = snapshot["job_ids"]
                save_record(path, record)
                break
            record.pop("last_quota_rejection_job_ids", None)
        waiting = [e["stage"] for e in record["experiments"]
                   if not record["jobs"].get(e["stage"], {}).get("job_id")]
        record["waiting_for_gpu_submission"] = waiting
        save_record(path, record)
        if waiting:
            code, root = record["code_root"], record["run_root"]
            number = sum(key.startswith("dispatch_") for key in record["jobs"]) + 1
            stage = f"dispatch_{number:04d}"
            command = ["sbatch", "--parsable", f"--chdir={code}",
                       f"--job-name=hs_abl_dispatch_{Path(root).name}",
                       f"--output={root}/logs/{stage}-%j.out",
                       f"--error={root}/logs/{stage}-%j.err", "--export=ALL",
                       "--begin=now" if number == 1 else "--begin=now+10minutes",
                       "scripts/slurm/37_dispatch_sharing_ablation.sh"]
            record["active_dispatcher"] = submit_stage(record, path, stage, command, environment)
            save_record(path, record)
            print(f"{len(waiting)} training jobs await capacity; "
                  f"controller {record['active_dispatcher']}")
        elif "summarize" not in record["jobs"]:
            submit_stage(record, path, "summarize", summary_command(record), environment)
            print("All training jobs and the dependent summary are submitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    dispatch_pending(args.receipt)


if __name__ == "__main__":
    main()
