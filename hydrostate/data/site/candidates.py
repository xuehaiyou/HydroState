"""Quality-screened morning candidates and strict-priority seeded station sampling."""

import json
import random
from bisect import bisect_left, insort
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from heapq import heapify, heappop, heapreplace
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked

from .archives import sha256
from .readers import UnsupportedSource, ismn_windows
from .temporal import DAY, epoch


def _read_candidate_file(args):
    row, root, config = args
    path = Path(row["source_path"])
    checkpoint = Path(root) / "labels" / (row["file_id"] + ".json")
    identity = fingerprint(
        dict(
            config=config,
            bytes=row["source_bytes"],
            mtime_ns=row["source_mtime_ns"],
            file=str(path),
        )
    )
    if (path.stat().st_size, path.stat().st_mtime_ns) != (
        row["source_bytes"],
        row["source_mtime_ns"],
    ):
        raise ValueError(f"Raw file changed after inventory: {path}")
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if result["identity"] != identity:
            raise ValueError("Candidate checkpoint definition changed")
    else:
        start = max(row["time_start"][:10], config["input_start"])
        # Include the last observed UTC date; window quality still uses the full window denominator.
        end = min(
            (pd.Timestamp(row["time_end"][:10]) + timedelta(days=1)).date().isoformat(),
            config["as_of"][:10],
        )
        try:
            labels = ismn_windows(path, start, end, config["as_of"]) if end > start else []
            valid = [r for r in labels if r["supervision_ready"]]
            result = dict(
                identity=identity,
                status="complete",
                windows=valid,
                temporal_windows=len(labels),
                valid_windows=len(valid),
            )
        except UnsupportedSource as error:
            result = dict(
                identity=identity,
                status="unsupported",
                reason=str(error),
                windows=[],
                temporal_windows=0,
                valid_windows=0,
            )
        atomic_json(checkpoint, result)
    candidates = []
    for i, record in enumerate(result["windows"]):
        candidates.append(
            dict(
                candidate_id=fingerprint([row["file_id"], record["target_time"]])[:24],
                file_id=row["file_id"],
                label_record=str(checkpoint),
                record_index=i,
                site_id=row["site_id"],
                physical_group_id=row["physical_group_id"],
                spatial_group_id=row["spatial_group_id"],
                split=row["split"],
                lon=row["lon"],
                lat=row["lat"],
                frequency_bin=int(row["frequency_bin"]),
                landcover=int(row["landcover"] or 0),
                climate_class=int(row["climate_class"] or 0),
                target_time=record["target_time"],
                target_end=record["target_end"],
                valid_days=record["sm"]["valid_days"],
                year=int(record["target_time"][:4]),
                month=int(record["target_time"][5:7]),
            )
        )
    return dict(
        file_id=row["file_id"],
        status=result["status"],
        valid_windows=result["valid_windows"],
        reason=result.get("reason"),
    ), candidates


def spaced_capacity(dates, gap_days=30):
    """Exact maximum count of sorted equal-gap timestamps; does not choose production samples."""
    last, count = -float("inf"), 0
    for day in sorted({epoch(v) for v in dates}):
        if day - last >= gap_days * DAY:
            count, last = count + 1, day
    return count


def prepare_candidates(index, splits, strata, output, *, as_of, workers=2, limit=None):
    index, splits, strata, output = map(Path, (index, splits, strata, output))
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    files = pd.read_parquet(index / "source_files.parquet")
    locations = pd.read_parquet(splits / "station_splits.parquet")
    spatial = pd.read_parquet(strata / "station_strata.parquet")
    properties = spatial[["location_id", "frequency_bin", "landcover", "climate_class"]]
    locations = locations.merge(properties, on="location_id", validate="one_to_one")
    eligible = files[files.sm_header_eligible.fillna(False)].merge(
        locations[
            [
                "site_id",
                "lon",
                "lat",
                "physical_group_id",
                "spatial_group_id",
                "split",
                "frequency_bin",
                "landcover",
                "climate_class",
            ]
        ],
        on=["site_id", "lon", "lat"],
        validate="many_to_one",
    )
    input_start = json.loads((strata / "strata_report.json").read_text())["input_start"]
    eligible = eligible[
        eligible.frequency_bin.ge(0)
        & eligible.split.ne("unresolved")
        & eligible.time_end.ge(input_start)
        & eligible.time_start.lt(as_of)
    ]
    if limit:
        eligible = eligible.head(limit)
    config = dict(
        version="site_sm_candidates_v1",
        as_of=as_of,
        limit=limit,
        input_start=input_start,
        source_sha256=sha256(index / "source_files.parquet"),
        split_sha256=sha256(splits / "station_splits.parquet"),
        strata_sha256=sha256(strata / "station_strata.parquet"),
    )
    with locked(output / ".candidates.lock", blocking=False):
        path = output / "candidate_config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Candidate inputs changed; use a new directory")
        atomic_json(path, config)
        tasks = [(r, str(output), config) for r in eligible.to_dict("records")]
        all_rows, status = [], []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (item, rows) in enumerate(pool.map(_read_candidate_file, tasks), 1):
                all_rows.extend(rows)
                status.append(item)
                if n % 50 == 0 or n == len(tasks):
                    print(
                        json.dumps(
                            dict(files_done=n, total=len(tasks), valid_sensor_windows=len(all_rows))
                        ),
                        flush=True,
                    )
        if not all_rows:
            atomic_json(
                output / "candidate_report.json", dict(complete=True, files=status, windows=0)
            )
            raise ValueError("No valid morning candidates")
        table = pd.DataFrame(all_rows)
        table.to_parquet(output / "candidates.parquet", index=False)
        counts = []
        for frequency in range(7):
            rows = table[table.frequency_bin.eq(frequency)]
            counts.append(
                dict(
                    frequency_bin=frequency,
                    physical_stations=rows.physical_group_id.nunique(),
                    sensor_windows=len(rows),
                    unique_station_windows=len(
                        rows.drop_duplicates(["physical_group_id", "target_time"])
                    ),
                    capacity_30d=sum(
                        spaced_capacity(g.target_time) for _, g in rows.groupby("physical_group_id")
                    ),
                    train_stations=rows[rows.split.eq("train")].physical_group_id.nunique(),
                    validation_stations=rows[
                        rows.split.eq("validation")
                    ].physical_group_id.nunique(),
                    test_stations=rows[rows.split.eq("test")].physical_group_id.nunique(),
                )
            )
        pd.DataFrame(counts).to_csv(output / "frequency_availability.csv", index=False)
        report = dict(
            complete=True,
            config=config,
            files=status,
            frequency=counts,
            windows=len(table),
            physical_stations=table.physical_group_id.nunique(),
            note="QC morning candidates; satellite availability not yet checked. Bin capacities "
            "are separate upper bounds when one colocated group spans multiple bins.",
        )
        atomic_json(output / "candidate_report.json", report)
        return report


class _SamplingBucket:
    """Lazy monotone priorities avoid rescanning every candidate after each selection."""

    def __init__(self, counter, key, *, children=(), rows=(), eligible=None):
        self.counter, self.key = counter, key
        self.rows, self.cursor, self.eligible = rows, 0, eligible
        self.heap = []
        for i, child in enumerate(children):
            choice = child.peek()
            if choice is not None:
                self.heap.append((choice[0], i, child))
        heapify(self.heap)

    def peek(self):
        prefix = () if self.counter is None else (self.counter[self.key],)
        if self.eligible is not None:
            while self.cursor < len(self.rows):
                rank, row = self.rows[self.cursor]
                if self.eligible(row):
                    return (*prefix, rank), row
                self.cursor += 1
            return None
        while self.heap:
            previous, index, child = self.heap[0]
            choice = child.peek()
            if choice is None:
                heappop(self.heap)
            elif choice[0] != previous:
                heapreplace(self.heap, (choice[0], index, child))
            else:
                return (*prefix, *choice[0]), choice[1]
        return None


class StratifiedPicker:
    """Water quotas, then land-cover×climate, year, month; site diversity breaks ties."""

    def __init__(self, rows, *, seed=43, gap_days=30, admissible=None):
        self.rows = sorted(rows, key=lambda r: r["candidate_id"])
        self.rng = random.Random(seed)
        self.rng.shuffle(self.rows)
        self.gap = gap_days * DAY
        self.counts = Counter()
        self.space, self.years, self.months, self.sites = (Counter() for _ in range(4))
        self.dates = defaultdict(list)
        self.attempted = set()
        self.admissible = admissible
        self.timestamps = {r["candidate_id"]: epoch(r["target_time"]) for r in self.rows}
        if len(self.timestamps) != len(self.rows):
            raise ValueError("Sampling candidate IDs must be unique")
        groups = defaultdict(list)
        for rank, row in enumerate(self.rows):
            groups[row["frequency_bin"]].append((rank, row))
        self.buckets = {b: self._build(rows) for b, rows in groups.items()}

    def _build(self, rows, level=0, key=None):
        counters = (None, self.space, self.years, self.months, self.sites)
        if level == 4:
            return _SamplingBucket(counters[level], key, rows=rows, eligible=self.eligible)
        groups = defaultdict(list)
        for rank, row in rows:
            b = row["frequency_bin"]
            space = (b, row["landcover"], row["climate_class"])
            keys = (
                space,
                (*space, row["year"]),
                (*space, row["year"], row["month"]),
                (b, row["physical_group_id"]),
            )
            groups[keys[level]].append((rank, row))
        return _SamplingBucket(
            counters[level],
            key,
            children=[
                self._build(members, level + 1, child_key) for child_key, members in groups.items()
            ],
        )

    def eligible(self, row):
        if row["candidate_id"] in self.attempted:
            return False
        dates = self.dates[row["physical_group_id"]]
        time = self.timestamps[row["candidate_id"]]
        i = bisect_left(dates, time)
        separated = (i == 0 or time - dates[i - 1] >= self.gap) and (
            i == len(dates) or dates[i] - time >= self.gap
        )
        return separated and (self.admissible is None or self.admissible(row))

    def next(self, quotas):
        # Choose the least fulfilled requested bin; exhausted bins cannot block the others.
        bins = sorted(
            (b for b in quotas if self.counts[b] < quotas[b]),
            key=lambda b: (self.counts[b] / quotas[b], b),
        )
        for b in bins:
            choice = self.buckets[b].peek() if b in self.buckets else None
            if choice is not None:
                return choice[1]
        return None

    def record(self, row, *, success):
        self.attempted.add(row["candidate_id"])
        if not success:
            return
        b = row["frequency_bin"]
        space = (b, row["landcover"], row["climate_class"])
        year = (*space, row["year"])
        self.counts[b] += 1
        self.space[space] += 1
        self.years[year] += 1
        self.months[(*year, row["month"])] += 1
        self.sites[(b, row["physical_group_id"])] += 1
        insort(self.dates[row["physical_group_id"]], self.timestamps[row["candidate_id"]])
