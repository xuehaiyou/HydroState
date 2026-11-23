"""Report one validation-selected checkpoint per manual-cover candidate."""

import argparse
import json
import math
import re
from pathlib import Path

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.manual_cover.contract import CLASSES

VARIANTS = ("CGC", "CoSA_Router", "B")


def compare_validation(models):
    """Compare observed errors and skill for the same selected epochs on frozen data."""
    pending = [
        name
        for name, result in models.items()
        if result.get("completed_validation_epoch", 0) < 6 or not result.get("checkpoint_present")
    ]
    if pending:
        return dict(status="awaiting_validation", pending=pending)

    glacier = "cover/glacier/all/"
    flood = "cover/sen1floods11/all/"
    minimize = ("glacier_mean_mae", "flood_water_mae", "sm_rmse", "et_rmse")
    maximize = ("glacier_accuracy", "flood_water_iou", "sm_r", "et_r")
    counts = [glacier + c + "_count" for c in CLASSES] + [
        flood + "water_count",
        "site/site_sm_count",
        "site/site_et_count",
    ]
    keys = {
        **{c + "_mae": glacier + c + "_mae" for c in CLASSES},
        "flood_water_mae": flood + "water_mae",
        "sm_rmse": "site/site_sm_rmse",
        "et_rmse": "site/site_et_rmse",
        "glacier_accuracy": glacier + "accuracy",
        "flood_water_iou": flood + "water_iou",
        "sm_r": "site/site_sm_correlation",
        "et_r": "site/site_et_correlation",
    }
    rows, supports = {}, {}
    for name, result in models.items():
        values = result["validation"]
        missing = [
            k
            for k in [*keys.values(), *counts]
            if values.get(k) is None or not math.isfinite(values[k])
        ]
        if missing:
            return dict(status="missing_validation_metrics", model=name, missing=missing)
        row = {k: values[v] for k, v in keys.items()}
        row["glacier_mean_mae"] = sum(row[c + "_mae"] for c in CLASSES) / len(CLASSES)
        rows[name] = row
        supports[name] = {k: values[k] for k in counts}
    support = next(iter(supports.values()))
    if min(support.values()) <= 0 or any(values != support for values in supports.values()):
        return dict(status="inspect_validation_support", counts=supports, values=rows)

    leaders = {}
    for metric in (*minimize, *maximize, *(c + "_mae" for c in CLASSES)):
        best = (max if metric in maximize else min)(row[metric] for row in rows.values())
        leaders[metric] = [name for name, row in rows.items() if row[metric] == best]
    return dict(
        status="development_point_estimates",
        values=rows,
        counts=support,
        leaders=leaders,
        best_on_four_errors=[n for n in rows if all(n in leaders[k] for k in minimize)],
        best_on_all_compared=[n for n in rows if all(n in names for names in leaders.values())],
    )


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
        key: value if value is not None and math.isfinite(value) else None
        for key, value in row.items()
        if key.startswith(("cover/", "site/", "product/"))
        or key in {"manual_cover_error", "manual_joint_score"}
    }


def test_rows(work):
    """Read the metric JSON dumped by MMEngine's standalone test loop."""
    rows = []
    for directory in sorted(Path(work).glob("*")):
        path = directory / f"{directory.name}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        score = row.get("manual_joint_score")
        if score is None or not math.isfinite(score):
            raise ValueError(f"Invalid final-test metric record: {path}")
        # MMEngine's test JSON drops numpy integer tags; the same run's text
        # log retains their exact integer values. Never round-trip float errors.
        missing_counts = [
            key for key, value in row.items()
            if value is None and key.startswith("cover/")
            and (key.endswith(("_count", "_support")) or "/confusion_" in key)
        ]
        if missing_counts:
            log_path = path.with_suffix(".log")
            lines = [
                line for line in log_path.read_text().splitlines()
                if "Epoch(test)" in line and "manual_joint_score:" in line
            ]
            if len(lines) != 1:
                raise ValueError(f"Expected one final-test metric log: {log_path}")
            values = dict(re.findall(r"(cover/\S+): (\d+)(?=\s|$)", lines[0]))
            for key in missing_counts:
                if key not in values:
                    raise ValueError(f"Missing exact integer {key} in {log_path}")
                row[key] = int(values[key])
            row["count_log"] = str(log_path)
        rows.append(dict(row, log=str(path)))
    return rows


def summarize(root, validation_only=False):
    root = Path(root)
    prepared_file = root / "prepared.json"
    prepared = json.loads(prepared_file.read_text()) if prepared_file.exists() else {}
    results = {}
    for name in prepared.get("variants", VARIANTS):
        directory = root / name
        rows = metric_rows(directory / "work")
        if not rows:
            results[name] = dict(status="no_validation_result")
            continue
        # Strict greater-is-better selection keeps the first result on ties.
        best = max(rows, key=lambda row: row["manual_joint_score"])
        epoch = int(best["step"])
        checkpoint = directory / "work" / f"best_manual_joint_score_epoch_{epoch}.pth"
        tests = [] if validation_only else test_rows(directory / "final_test")
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
            test_count_log=tests[0].get("count_log") if tests else None,
        )
        result["status"] = "incomplete"
        if result["completed_validation_epoch"] >= 6 and result["checkpoint_present"]:
            if validation_only:
                result["status"] = "validation_available"
            elif tests:
                result["status"] = "metrics_available"
        results[name] = result
    return dict(
        scope="One manual_joint_score-selected checkpoint per model; validation and test separate. "
        "Metric availability alone does not establish successful Slurm completion.",
        glacier_interpretation="point assumed pure at 30m; not measured subpixel area",
        flood_interpretation="raster-derived water area fraction; no land/ice truth",
        prepared=prepared or None,
        validation_comparison=compare_validation(results),
        models=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-only", action="store_true", help="Skip all final-test logs")
    args = parser.parse_args()
    report = summarize(args.root, validation_only=args.validation_only)
    if args.output:
        atomic_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
