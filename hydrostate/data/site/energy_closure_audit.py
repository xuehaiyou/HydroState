"""Read-only EBR audit: Mauder (2024) Eq.2; daytime sensitivity Gan (2018).

ONEFlux mapping and record-span handling are explicit project adaptations, not
claims of exact PML replication. No annual/quarterly coverage cutoff is applied.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "LE_F_MDS",
    "H_F_MDS",
    "G_F_MDS",
    "NETRAD",
    "SW_IN_F_MDS",
    "SW_OUT",
    "LW_IN_F_MDS",
    "LW_OUT",
]
COLS = FIELDS + [f"{x}_QC" for x in FIELDS] + ["TIMESTAMP_START", "TIMESTAMP_END"]


def energy_terms(frame, good_fill=False):
    """Return common finite/quality mask, available and turbulent energy rates."""

    def field(name):
        return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)

    def good(name):
        value = field(name)
        # Some measured radiation fields have no separate ONEFlux HH quality flag.
        return np.isfinite(value) & (
            field(name + "_QC").isin([0, 1] if good_fill else [0])
            if name + "_QC" in frame
            else True
        )

    net = field("NETRAD")
    direct = good("NETRAD")
    radiation = ["SW_IN_F_MDS", "SW_OUT", "LW_IN_F_MDS", "LW_OUT"]
    reconstructed = (
        field(radiation[0]) - field(radiation[1]) + field(radiation[2]) - field(radiation[3])
    )
    reconstructed_good = np.logical_and.reduce([good(c) for c in radiation])
    rn = net.where(direct, reconstructed)
    valid = (direct | reconstructed_good) & good("LE_F_MDS") & good("H_F_MDS") & good("G_F_MDS")
    # For gap-filled energy fields, a missing QC column cannot establish measured status.
    for name in ["LE_F_MDS", "H_F_MDS", "G_F_MDS"]:
        if name + "_QC" not in frame:
            valid &= False
    return (
        valid.to_numpy(copy=True),
        (rn - field("G_F_MDS")).to_numpy(),
        (field("LE_F_MDS") + field("H_F_MDS")).to_numpy(),
    )


def aggregate(frame, good_fill=False, daytime=False):
    valid, available, turbulent = energy_terms(frame, good_fill)
    seconds = (frame.end - frame.start).dt.total_seconds().to_numpy()
    valid &= (seconds > 0) & (seconds <= 3600)
    if daytime:
        valid &= frame.get("SW_IN_F_MDS", pd.Series(np.nan, index=frame.index)).to_numpy() > 20
        if "SW_IN_F_MDS_QC" in frame:
            valid &= frame.SW_IN_F_MDS_QC.isin([0, 1] if good_fill else [0]).to_numpy()
    den = float(np.sum(available[valid] * seconds[valid]))
    num = float(np.sum(turbulent[valid] * seconds[valid]))
    ebr = num / den if den > 0 else None
    dates = frame.loc[valid, "start"]
    # Span is reported separately: it is not a claim of uninterrupted observations.
    span = float((dates.max() - dates.min()).total_seconds() / 86400) if len(dates) else 0.0
    return dict(
        ebr=ebr,
        numerator_j_m2=num,
        denominator_j_m2=den,
        valid_intervals=int(valid.sum()),
        valid_hours=float(seconds[valid].sum() / 3600),
        common_span_days=span,
        monthly_intervals=dates.dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
        status="insufficient_record_span"
        if span <= 365.25
        else ("unassessable_energy" if ebr is None else ("pass" if ebr > 0.75 else "fail")),
    )


def audit_station(item):
    station, records = item
    counts = Counter(r["source_file"]["path"] for r in records if "source_file" in r)
    if not counts:
        return dict(station=station, error="No HH/HR source recorded")
    frames = []
    for path in sorted(counts, key=lambda p: (-counts[p], p)):
        f = pd.read_csv(
            path,
            usecols=lambda c: c in COLS,
            na_values=[-9999],
            dtype={"TIMESTAMP_START": str, "TIMESTAMP_END": str},
        )
        f["start"] = pd.to_datetime(f.TIMESTAMP_START, format="%Y%m%d%H%M")
        f["end"] = pd.to_datetime(f.TIMESTAMP_END, format="%Y%m%d%H%M")
        f = f[(f.start >= "2015-01-01") & (f.end <= "2026-01-01")]
        if f.start.duplicated().any():
            raise ValueError(f"Duplicate timestamps within {path}")
        frames.append(f)
    # Prioritise most-used published source on overlaps, before inspecting EBR.
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("start").sort_values("start")
    if (frame.start.iloc[1:].to_numpy() < frame.end.iloc[:-1].to_numpy()).any():
        raise ValueError(f"Overlapping intervals: {station}")
    results = {}
    for fill in [False, True]:
        for day in [False, True]:
            name = ("measured_goodfill" if fill else "measured") + ("_day" if day else "_all")
            results[name] = aggregate(frame, fill, day)
    return dict(station=station, sources=list(counts), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.records.read_text())
    groups = {}
    for row in rows:
        groups.setdefault(row["station"], []).append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    stations = []
    with ThreadPoolExecutor(2) as pool:
        for station in pool.map(audit_station, groups.items()):
            stations.append(station)
            (args.output / "stations.json").write_text(json.dumps(stations, indent=2))
            print(f"audited {len(stations)}/{len(groups)} {station['station']}", flush=True)
    summary = {}
    for mode in ["measured_all", "measured_day", "measured_goodfill_all", "measured_goodfill_day"]:
        status = {
            s["station"]: s.get("results", {}).get(mode, {}).get("status", "no_source")
            for s in stations
        }
        split_counts = {}
        for split in ["all", "train", "validation", "test"]:
            subset = [r for r in rows if split == "all" or r["split"] == split]
            keep = [r for r in subset if status[r["station"]] == "pass"]
            split_counts[split] = dict(
                total=len(subset),
                kept=len(keep),
                stations=len({r["station"] for r in keep}),
                status_windows=dict(Counter(status[r["station"]] for r in subset)),
                frequency=dict(Counter(r["frequency_bin"] for r in keep)),
            )
        summary[mode] = dict(station_status=dict(Counter(status.values())), splits=split_counts)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
