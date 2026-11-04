"""Reproducible stratified candidate draws; satellite acceptance is a later stage."""

import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .availability import availability_rows
from .candidates import StratifiedPicker, spaced_capacity
from .temporal import DAY, epoch

IDENTITY = ["physical_group_id", "lon", "lat", "target_time", "target_end"]


class MaximumDateSchedule:
    """Keep only choices belonging to a maximum 30-day-separated station schedule.

    Prefix/suffix greedy capacities are exact for equal date gaps. Once a feasible
    choice is fixed, a date outside every maximum schedule can never become feasible;
    this monotonicity also permits the picker's lazy removal of inadmissible dates.
    """

    def __init__(self, table, gap_days):
        if gap_days <= 0:
            raise ValueError("Positive station date gap required")
        if table.groupby("physical_group_id").frequency_bin.nunique().gt(1).any():
            raise ValueError("Cross-bin physical stations require joint quota resolution")
        self.gap = gap_days * DAY
        self.times, self.segments, self.allowed = {}, {}, set()
        for site, rows in table.groupby("physical_group_id", sort=False):
            times = np.array(sorted({epoch(t) for t in rows.target_time}), dtype="int64")
            self.times[site] = times
            self._partition(site, 0, len(times))

    def _partition(self, site, left, right):
        times = self.times[site][left:right]
        if not len(times):
            return
        early, late = [], []
        i = 0
        while i < len(times):
            early.append(times[i])
            i = int(np.searchsorted(times, times[i] + self.gap))
        i = len(times) - 1
        while i >= 0:
            late.append(times[i])
            i = int(np.searchsorted(times, times[i] - self.gap, side="right")) - 1
        before = np.searchsorted(early, times - self.gap, side="right")
        after = len(late) - np.searchsorted(late[::-1], times + self.gap)
        feasible = before + 1 + after == len(early)
        for time, valid in zip(times, feasible, strict=True):
            key = (site, int(time))
            self.segments[key] = (left, right)
            self.allowed.discard(key)
            if valid:
                self.allowed.add(key)

    def eligible(self, row):
        return (row["physical_group_id"], epoch(row["target_time"])) in self.allowed

    def select(self, row):
        site, time = row["physical_group_id"], epoch(row["target_time"])
        if (site, time) not in self.allowed:
            raise ValueError("Selection would reduce feasible water-group capacity")
        left, right = self.segments[(site, time)]
        times = self.times[site]
        before = max(left, int(np.searchsorted(times, time - self.gap, side="right")))
        after = min(right, int(np.searchsorted(times, time + self.gap)))
        for value in times[before:after]:
            self.allowed.discard((site, int(value)))
        self._partition(site, left, before)
        self._partition(site, after, right)


def bundle_candidates(tables):
    """Bundle only identical coordinates/time; nearby sensors keep their actual centres.

    All original label references survive in members. This is an inventory of potential
    input samples, not a merge of values or a claim that ET support is already approved.
    """
    available = [t for t in tables if not t.empty]
    if not available:
        return pd.DataFrame(), pd.DataFrame()
    members = pd.concat(available, ignore_index=True)
    members["input_candidate_id"] = [
        fingerprint(list(v))[:24] for v in members[IDENTITY].itertuples(index=False, name=None)
    ]
    groups = members.groupby("input_candidate_id", sort=False)
    if groups[["split", "frequency_bin", "landcover", "climate_class"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting split/strata at identical station coordinates and time")
    if members.groupby("physical_group_id").split.nunique().gt(1).any():
        raise ValueError("Physical station crosses fixed splits")
    members["has_sm"] = members.target.eq("sm")
    members["has_et"] = members.target.eq("et")
    members["has_et_footprint"] = members.has_et & members.operator_ready
    members["source_priority"] = members.source.map({"fluxnet": 0, "icos": 1, "ismn": 2})
    if members.source_priority.isna().any():
        raise ValueError("Unregistered source priority")
    # Prefer supported operators for the representative; preserve every alternative reference.
    ordered = members.sort_values(
        ["operator_ready", "source_priority", "candidate_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    preferred = ordered.drop_duplicates("input_candidate_id").copy()
    counts = groups[["has_sm", "has_et", "has_et_footprint"]].max()
    preferred = preferred.drop(columns=["has_sm", "has_et", "has_et_footprint"]).merge(
        counts, on="input_candidate_id", validate="one_to_one"
    )
    preferred["representative_label_id"] = preferred.candidate_id
    preferred["candidate_id"] = preferred.input_candidate_id
    preferred["satellite_inputs_checked"] = False
    members = members.drop(columns=["source_priority", "has_sm", "has_et", "has_et_footprint"])
    return preferred.sort_values("candidate_id").reset_index(drop=True), members


def draw_capacity(table, *, seed=43, gap_days=30):
    if table.empty:
        return table.copy(), dict(quotas={}, selected={}, unknown_or_unresolved=0)
    eligible = table[
        table.frequency_bin.between(0, 6) & table.split.isin(["train", "validation", "test"])
    ].copy()
    quotas = {
        b: sum(
            spaced_capacity(g.target_time, gap_days)
            for _, g in eligible[eligible.frequency_bin.eq(b)].groupby("physical_group_id")
        )
        for b in range(7)
    }
    schedule = MaximumDateSchedule(eligible, gap_days)
    picker = StratifiedPicker(
        eligible.to_dict("records"), seed=seed, gap_days=gap_days, admissible=schedule.eligible
    )
    selected = []
    while (row := picker.next(quotas)) is not None:
        selected.append(dict(row, draw_order=len(selected)))
        schedule.select(row)
        picker.record(row, success=True)
    chosen = pd.DataFrame(selected) if selected else eligible.iloc[:0].copy()
    # Independently audit the actual sequence; capacities are not claims of optimal random draws.
    for _, group in chosen.groupby("physical_group_id"):
        dates = sorted(epoch(v) for v in group.target_time)
        if any(b - a < gap_days * DAY for a, b in zip(dates, dates[1:])):
            raise ValueError("Sample selection violates global per-station date separation")
    if any(picker.counts[b] != quotas[b] for b in quotas):
        raise ValueError("Sampling did not preserve the available frequency quotas")
    return chosen, dict(
        quotas=quotas,
        selected={b: picker.counts[b] for b in quotas},
        unknown_or_unresolved=len(table) - len(eligible),
        method="maximum_capacity_preserving_seeded_landcover_climate_year_month_draw",
    )


def write_sampling_plan(table, members, output, *, seed, gap_days):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / "input_candidates.parquet", index=False)
    members.to_parquet(output / "label_members.parquet", index=False)
    chosen, report = draw_capacity(table, seed=seed, gap_days=gap_days)
    chosen.to_parquet(output / "selected.parquet", index=False)
    counts = report["selected"]
    active = [b for b, n in counts.items() if n > 0]
    equal_n = min((counts[b] for b in active), default=0)
    balanced = chosen.groupby("frequency_bin", sort=False).head(equal_n) if len(chosen) else chosen
    balanced.to_parquet(output / "equal_nonempty_bins.parquet", index=False)
    summaries = []
    for stage, frame in (
        ("candidate", table),
        ("random_draw", chosen),
        ("equal_nonempty", balanced),
    ):
        summaries.extend(availability_rows(frame, stage))
    pd.DataFrame(summaries).to_csv(output / "counts.csv", index=False)
    if not chosen.empty:
        chosen.groupby(
            ["split", "frequency_bin", "landcover", "climate_class", "year", "month"]
        ).size().rename("samples").reset_index().to_csv(output / "strata_counts.csv", index=False)
    report.update(
        input_candidates=len(table),
        selected_total=len(chosen),
        selected_physical_stations=int(chosen.physical_group_id.nunique()) if len(chosen) else 0,
        selected_by_split=chosen.split.value_counts().to_dict() if len(chosen) else {},
        selected_labels={
            name: int(chosen[name].sum()) if len(chosen) else 0
            for name in ("has_sm", "has_et", "has_et_footprint")
        },
        equal_nonempty_bins=active,
        equal_nonempty_per_bin=equal_n,
        equal_nonempty_total=len(balanced),
        strict_seven_equal_per_bin=min(counts.values(), default=0),
        frequency=summaries,
        satellite_validated_samples=0,
        scope="Planning draws of label candidates; no satellite download or training release",
    )
    atomic_json(output / "report.json", report)
    return report


def sample_capacity(
    sm_candidates, et_candidates, footprint_candidates, output, *, seed=43, gap_days=30
):
    paths = list(map(Path, (sm_candidates, et_candidates, footprint_candidates)))
    quality_report = json.loads((paths[1].parent / "report.json").read_text())
    if not quality_report.get("complete") or quality_report.get("candidates_sha256") != sha256(
        paths[1]
    ):
        raise ValueError("Sampling requires the complete, verified ET temporal candidate manifest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version="site_stratified_capacity_v1",
        seed=seed,
        gap_days=gap_days,
        priority=["water_frequency", "land_cover_type_x_climate_zone", "year", "month", "site_tie"],
        input_identity=IDENTITY,
        sources_sha256={str(p): sha256(p) for p in paths},
    )
    with locked(output / ".sampling.lock", blocking=False):
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Sampling inputs or rules changed; use a new output directory")
        if (output / "report.json").exists():
            return json.loads((output / "report.json").read_text())
        atomic_json(config_path, config)
        sm, et, footprint = [pd.read_parquet(p) for p in paths]
        sm = sm.assign(target="sm", source="ismn", operator_ready=True)
        et = et.assign(target="et", operator_ready=False)
        footprint = footprint.assign(target="et", source="icos", operator_ready=True)
        cohorts = dict(
            et_temporal=[et], joint_temporal=[sm, et, footprint], joint_supported=[sm, footprint]
        )
        result = dict(complete=False, config=config, cohorts={})
        for name, tables in cohorts.items():
            began = monotonic()
            candidates, members = bundle_candidates(tables)
            report = write_sampling_plan(
                candidates, members, output / name, seed=seed, gap_days=gap_days
            )
            result["cohorts"][name] = {k: v for k, v in report.items() if k != "frequency"}
            print(
                json.dumps(
                    dict(
                        cohort=name, elapsed_seconds=monotonic() - began, **result["cohorts"][name]
                    )
                ),
                flush=True,
            )
        result["complete"] = True
        atomic_json(output / "report.json", result)
        return result
