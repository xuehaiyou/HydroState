"""Report one validation-selected checkpoint per manual-cover candidate."""

import argparse
import json
import math
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json

VARIANTS = ("CGC", "CoSA_Router", "B")


def metric_rows(work):
    rows = []
    for path in sorted(Path(work).glob("*/vis_data/scalars.json")):
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            score = row.get("manual_joint_score")
            if score is not None and math.isfinite(score):
                rows.append(dict(row, log=str(path)))
    return rows


def metrics(row):
    return {
        key: value
        for key, value in row.items()
        if key.startswith(("cover/", "site/", "product/"))
        or key in {"manual_cover_error", "manual_joint_score"}
    }


def summarize(root):
    root = Path(root)
    results = {}
    for name in VARIANTS:
        directory = root / name
        rows = metric_rows(directory / "work")
        if not rows:
            results[name] = dict(status="no_validation_result")
            continue
        # Strict greater-is-better selection keeps the first result on ties.
        best = max(rows, key=lambda row: row["manual_joint_score"])
        epoch = int(best["step"])
        checkpoint = directory / "work" / f"best_manual_joint_score_epoch_{epoch}.pth"
        tests = metric_rows(directory / "final_test")
        if len(tests) > 1:
            raise ValueError(f"Multiple final-test results for {name}; inspect provenance")
        result = dict(
            selected_epoch=epoch,
            completed_validation_epoch=max(int(row["step"]) for row in rows),
            checkpoint=str(checkpoint),
            checkpoint_present=checkpoint.is_file(),
            validation_log=best["log"],
            validation=metrics(best),
            test=metrics(tests[0]) if tests else None,
            test_log=tests[0]["log"] if tests else None,
        )
        result["status"] = (
            "metrics_available"
            if result["completed_validation_epoch"] >= 6 and result["checkpoint_present"] and tests
            else "incomplete"
        )
        results[name] = result
    prepared = root / "prepared.json"
    return dict(
        scope="One manual_joint_score-selected checkpoint per model; validation and test separate. "
        "Metric availability alone does not establish successful Slurm completion.",
        glacier_interpretation="point assumed pure at 30m; not measured subpixel area",
        flood_interpretation="raster-derived water area fraction; no land/ice truth",
        prepared=json.loads(prepared.read_text()) if prepared.exists() else None,
        models=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    if args.output:
        atomic_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
