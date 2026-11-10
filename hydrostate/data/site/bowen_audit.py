"""Read-only provider Bowen-label inventory; never changes production labels.

Twine's Bowen-ratio assumption is distinct from ONEFlux's implementation.
This audit uses supplied LE_CORR, not an unvalidated replacement algorithm.
Local-day coverage and UTC target support are kept separate.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import atomic_json

from .et_candidates import indexed_flux_files
from .temporal import DAY, epoch, interval_et, windows

COVERAGE = 0.75
MEASURED = 0.5
FIELDS = {
    "TIMESTAMP_START",
    "TIMESTAMP_END",
    "LE_F_MDS",
    "LE_F_MDS_QC",
    "LE_CORR",
    "H_CORR",
    "H_F_MDS",
    "H_F_MDS_QC",
    "NETRAD",
    "G_F_MDS",
    "G_F_MDS_QC",
}


def split_local_days(frame):
    """Split midnight-crossing HH/HR supports without changing interval means."""
    frame = frame.copy()
    for name, target in [("TIMESTAMP_START", "local_start"), ("TIMESTAMP_END", "local_end")]:
        frame[target] = (
            pd.to_datetime(frame[name], format="%Y%m%d%H%M").astype("datetime64[s]").astype("int64")
        )
    frame = frame.sort_values("local_start").reset_index(drop=True)
    if frame.local_start.duplicated().any():
        raise ValueError("Duplicate timestamps require explicit source resolution")
    starts, ends = frame.local_start.to_numpy(), frame.local_end.to_numpy()
    if np.any(ends <= starts) or np.any(ends - starts > 3600) or np.any(ends[:-1] > starts[1:]):
        raise ValueError("Invalid or overlapping HH/HR supports")
    boundary = (starts // DAY + 1) * DAY
    crossing = ends > boundary
    if crossing.any():
        tails = frame.loc[crossing].copy()
        tails["local_start"] = boundary[crossing]
        frame.loc[crossing, "local_end"] = boundary[crossing]
        frame = pd.concat([frame, tails]).sort_values("local_start").reset_index(drop=True)
    return frame


def daily_support(starts, ends, valid, coverage=COVERAGE):
    """Coverage of complete local-standard days, including absent records."""
    days, inverse = np.unique(starts // DAY, return_inverse=True)
    seconds = np.bincount(inverse, weights=(ends - starts) * valid)
    accepted = seconds >= coverage * DAY
    return valid & accepted[inverse], dict(days=len(days), accepted_days=int(accepted.sum()))


def prepare(path, offset):
    frame = pd.read_csv(
        path,
        usecols=lambda c: c in FIELDS,
        na_values=[-9999],
        dtype={"TIMESTAMP_START": str, "TIMESTAMP_END": str},
    )
    required = {"TIMESTAMP_START", "TIMESTAMP_END", "LE_F_MDS", "LE_F_MDS_QC"}
    if not required <= set(frame):
        raise ValueError("Missing verified ONEFlux HH/HR fields")
    frame = split_local_days(frame)
    start, end = frame.local_start.to_numpy(), frame.local_end.to_numpy()
    qc = frame.LE_F_MDS_QC.to_numpy()
    raw = frame.LE_F_MDS.to_numpy()
    corrected = frame.get("LE_CORR", pd.Series(np.nan, index=frame.index)).to_numpy()
    raw_good = np.isfinite(raw) & np.isin(qc, [0, 1])
    corrected_good = raw_good & np.isfinite(corrected)
    raw_mask, raw_days = daily_support(start, end, raw_good)
    corrected_mask, corrected_days = daily_support(start, end, corrected_good)
    result = dict(
        starts=start - int(offset * 3600),
        ends=end - int(offset * 3600),
        raw=raw,
        corrected=corrected,
        qc=qc,
        raw_mask=raw_mask,
        corrected_mask=corrected_mask,
    )
    # Availability for a possible future ONEFlux reconstruction, NOT eligible labels.
    required_energy = {"H_F_MDS", "H_F_MDS_QC", "NETRAD", "G_F_MDS", "G_F_MDS_QC"}
    energy = np.zeros(len(frame), bool)
    if required_energy <= set(frame):
        energy = (
            raw_good
            & np.isfinite(frame.H_F_MDS)
            & frame.H_F_MDS_QC.isin([0, 1])
            & np.isfinite(frame.NETRAD)
            & np.isfinite(frame.G_F_MDS)
            & frame.G_F_MDS_QC.eq(0)
        ).to_numpy()
    ratio_check = {}
    if {"H_CORR", "H_F_MDS"} <= set(frame):
        h, hc = frame.H_F_MDS.to_numpy(), frame.H_CORR.to_numpy()
        mask = corrected_good & np.isfinite(h) & np.isfinite(hc)
        numerator = hc[mask] * raw[mask] - h[mask] * corrected[mask]
        denominator = np.abs(hc[mask] * raw[mask]) + np.abs(h[mask] * corrected[mask])
        stable = denominator > 1.0
        error = np.abs(numerator[stable]) / denominator[stable]
        ratio_check = dict(
            intervals=int(stable.sum()),
            relative_cross_product_median=float(np.median(error)) if len(error) else None,
            relative_cross_product_p95=float(np.quantile(error, 0.95)) if len(error) else None,
        )
    info = dict(
        columns=sorted(frame.columns),
        raw_days=raw_days,
        corrected_days=corrected_days,
        corrected_intervals=int(corrected_good.sum()),
        raw_intervals=int(raw_good.sum()),
        potential_energy_intervals=int(energy.sum()),
        missing_corrected_with_energy=int((energy & ~corrected_good).sum()),
        bowen_preservation=ratio_check,
    )
    return result, info


def label_window(data, meta, left, right):
    a = np.searchsorted(data["ends"], epoch(left), side="right")
    b = np.searchsorted(data["starts"], epoch(right))
    kwargs = dict(
        latitude=meta["lat"],
        longitude=meta["lon"],
        start=left,
        end=right,
        min_valid=0,
        min_measured=MEASURED,
        min_day_night=COVERAGE,
    )

    def aggregate(values, mask):
        return interval_et(
            data["starts"][a:b],
            data["ends"][a:b],
            data[values][a:b],
            data["qc"][a:b],
            support_valid=data[mask][a:b],
            **kwargs,
        )

    raw = aggregate("raw", "raw_mask")
    corrected = aggregate("corrected", "corrected_mask")
    matched = aggregate("raw", "corrected_mask")
    row = dict(
        target_time=left,
        target_end=right,
        raw_valid=raw["valid"],
        corrected_valid=corrected["valid"],
        raw_value=raw["value"],
        corrected_value=corrected["value"],
        raw_matched_value=matched["value"],
        raw_reasons=",".join(raw["reasons"]),
        corrected_reasons=",".join(corrected["reasons"]),
    )
    for name in ["valid_fraction", "measured_fraction", "day_fraction", "night_fraction"]:
        row["corrected_" + name] = corrected[name]
    return row


def audit_file(task):
    meta, config, output, requested = task
    began = monotonic()
    out = Path(output)
    file_id = meta["file_id"]
    data, info = prepare(meta["source_path"], meta["utc_offset_hours"])
    offset = pd.Timedelta(hours=meta["utc_offset_hours"])
    first = (pd.Timestamp(meta["time_start"]) - offset - pd.Timedelta(days=7)).date().isoformat()
    last = (pd.Timestamp(meta["time_end"]) - offset + pd.Timedelta(days=8)).date().isoformat()
    left, right = max(config["input_start"], first), min(config["as_of"][:10], last)
    rows = [label_window(data, meta, a, b) for a, b in windows(left, right, as_of=config["as_of"])]
    table = pd.DataFrame(rows)
    identity = {
        k: meta[k]
        for k in [
            "file_id",
            "site_id",
            "source",
            "physical_group_id",
            "split",
            "frequency_bin",
            "landcover",
            "climate_class",
        ]
    }
    for key, value in identity.items():
        table[key] = value
    table.to_parquet(out / "windows" / (file_id + ".parquet"), index=False)
    # Exact old pairs retain their dates/locations and use common raw/corrected supports.
    checks = []
    for r in requested:
        check = label_window(data, meta, r["target_time"], r["target_end"])
        checks.append(
            dict(
                check, sample_id=r["sample_id"], group_id=meta["physical_group_id"], file_id=file_id
            )
        )
    summary = dict(
        identity,
        **info,
        calendar_windows=len(rows),
        raw_windows=int(table.raw_valid.sum()) if len(rows) else 0,
        corrected_windows=int(table.corrected_valid.sum()) if len(rows) else 0,
        raw_only_windows=int((table.raw_valid & ~table.corrected_valid).sum()) if len(rows) else 0,
        source_path=meta["source_path"],
        elapsed_seconds=monotonic() - began,
        requested_checks=checks,
    )
    atomic_json(out / "files" / (file_id + ".json"), summary)
    return summary


def build_tasks(base, output):
    base = Path(base)
    config = json.loads((base / "et_candidates_20261104/config.json").read_text())
    files = indexed_flux_files(
        base / "index_v2_20261103",
        base / "splits_20261103",
        base / "strata_20261103",
        ["fluxnet", "icos"],
    )
    # Reuse the established input-period and metadata eligibility, including zero-pass files.
    admitted = {p.stem for p in (base / "et_candidates_20261104/labels").glob("*.json")}
    files = files[files.file_id.isin(admitted)].copy()
    requested = json.loads((base / "reports/et_correction_20261110/selection.json").read_text())[
        "selected"
    ]
    lookup = {}
    for r in requested:
        lookup.setdefault(r["source_file"]["path"], []).append(r)
    records = files.to_dict("records")
    # Finish the small matched validation first, without reading those files a second time.
    records.sort(key=lambda r: (r["source_path"] not in lookup, r["source_bytes"], r["file_id"]))
    return [(r, config, str(output), lookup.get(r["source_path"], [])) for r in records]


def summarize(output, tasks, summaries):
    tables = [
        pd.read_parquet(Path(output) / "windows" / (r["file_id"] + ".parquet")) for r in summaries
    ]
    all_rows = pd.concat(tables, ignore_index=True)
    all_rows.to_parquet(Path(output) / "all_windows.parquet", index=False)
    good = all_rows[all_rows.corrected_valid].copy()
    # Count physical station-window identities once across sources; not source-row totals.
    unique = good.drop_duplicates(["physical_group_id", "target_time", "target_end"])
    report = dict(
        complete=len(summaries) == len(tasks),
        files=len(summaries),
        raw_candidate_windows=int(all_rows.raw_valid.sum()),
        corrected_source_windows=len(good),
        corrected_unique_windows=len(unique),
        corrected_sites=int(unique.physical_group_id.nunique()),
        files_with_any_corrected=sum(r["corrected_intervals"] > 0 for r in summaries),
        files_without_corrected=sum(r["corrected_intervals"] == 0 for r in summaries),
        files_missing_corrected_with_energy=sum(
            r["missing_corrected_with_energy"] > 0 for r in summaries
        ),
        frequency=unique.frequency_bin.value_counts().sort_index().to_dict(),
        source=good.source.value_counts().to_dict(),
        qc=dict(
            local_day=COVERAGE,
            daylight=COVERAGE,
            night=COVERAGE,
            measured=MEASURED,
            whole_window="diagnostic_only; implied by day/night coverage",
            ebr_site_gate=False,
        ),
        scope=(
            "Temporal candidates only; spatial/input/split checks still required. "
            "No new self-correction."
        ),
        failures=dict(Counter(x for s in all_rows.corrected_reasons for x in s.split(",") if x)),
    )
    atomic_json(Path(output) / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["files", "windows"]:
        (args.output / name).mkdir(exist_ok=True)
    tasks = build_tasks(args.base, args.output)
    if args.limit:
        tasks = tasks[: args.limit]
    atomic_json(
        args.output / "protocol.json",
        dict(
            coverage=COVERAGE,
            measured=MEASURED,
            source="provider LE_CORR",
            tasks=len(tasks),
            production_modified=False,
        ),
    )
    complete, pending = [], []
    for task in tasks:
        checkpoint = args.output / "files" / (task[0]["file_id"] + ".json")
        if checkpoint.exists():
            complete.append(json.loads(checkpoint.read_text()))
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(audit_file, pending, chunksize=1):
            complete.append(result)
            atomic_json(args.output / "progress.json", dict(done=len(complete), total=len(tasks)))
            print(
                json.dumps(
                    dict(
                        file=result["file_id"],
                        done=len(complete),
                        total=len(tasks),
                        seconds=round(result["elapsed_seconds"], 2),
                    )
                ),
                flush=True,
            )
    summarize(args.output, tasks, complete)


if __name__ == "__main__":
    main()
