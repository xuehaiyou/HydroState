"""Resumable bounded collection: centre proposals -> validated immutable shards."""

import json
import logging
import os
import random
import shutil
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.gee.products import BANDS
from hydrostate.storage import SampleShardWriter

from .catalog import candidate_blocks, catalog, pml_calendar
from .composites import QUALITY, normalize, sensor_composite
from .contract import PROFILE, VERSION, RetrievalSchema, intervals, product_end
from .gee_labels import fetch_cover, fetch_products
from .labels import build_targets, cover_targets
from .sampling import SamplingLedger, location_key, quotas
from .validation import validate_arrays

LOG = logging.getLogger(__name__)


def epoch(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def input_arrays(ee, sample, cache, norm, config):
    schema = RetrievalSchema()
    arrays = {name: np.zeros(spec.tail_shape, spec.dtype) for name, spec in schema.inputs.items()}
    slots = intervals(sample["target_time"])
    arrays["target_bounds"][:] = [epoch(sample["target_time"]), epoch(sample["target_end"])]
    arrays["slot_bounds"][:] = [[epoch(a), epoch(b)] for a, b in slots]
    catalogs = {
        sensor: catalog(ee, cache, sensor, sample, slots[0][0], slots[-1][1]) for sensor in BANDS
    }
    # Choose one relative orbit AND pass for all radar slots. Coverage, then
    # deterministic identity breaks ties; missing radar never rejects good optics.
    orbit_counts = Counter(
        (s.get("relativeOrbitNumber_start"), s.get("orbitProperties_pass")) for s in catalogs["s1"]
    )
    orbit = (
        min(orbit_counts, key=lambda key: (-orbit_counts[key], str(key))) if orbit_counts else None
    )
    scenes = {}
    for slot, (start, end) in enumerate(slots):
        for modality, sensor in enumerate(BANDS):
            selected = [
                s
                for s in catalogs[sensor]
                if epoch(start) * 1000 <= s["system:time_start"] < epoch(end) * 1000
                and (
                    sensor != "s1"
                    or (s.get("relativeOrbitNumber_start"), s.get("orbitProperties_pass")) == orbit
                )
            ]
            ids = [s["asset_id"] for s in selected]
            scenes[f"{sensor}_{slot}"] = ids
            if not ids:
                continue
            names = [*BANDS[sensor], *QUALITY]
            expression = sensor_composite(ee, sensor, ids, start)
            raw = compute_pixels(
                ee,
                expression,
                names,
                sample["crs"],
                sample["transform"],
                schema.width,
                schema.height,
            )
            valid = raw["days"] > 0
            arrays[sensor][slot] = normalize(
                sensor, np.stack([raw[name] for name in BANDS[sensor]]), valid, norm
            )
            arrays[f"{sensor}_valid"][slot, 0] = valid
            arrays[f"{sensor}_quality"][slot] = np.stack([raw[name] for name in QUALITY])
            arrays["sensor_valid"][slot, modality] = valid.any()
            # A structural slot reference, NOT a fabricated acquisition timestamp.
            # Actual date support remains in per-pixel quality and scene IDs.
            arrays["observation_time"][slot, modality] = (epoch(start) + epoch(end)) // 2
    active = sum(
        any(
            arrays[f"{sensor}_valid"][slot].mean() >= config["min_input_fraction"]
            for sensor in BANDS
        )
        for slot in range(schema.time_steps)
    )
    return arrays, scenes, active >= config["min_active_slots"]


def collect_sample(ee, sample, root, config, norm, calendar):
    sample = dict(
        sample,
        target_end=product_end(sample["target_time"]),
        version=VERSION,
        input_profile=PROFILE,
    )
    sample["sample_id"] = fingerprint(sample)[:24]
    sid = sample["sample_id"]
    rejection = root / "rejected" / f"{sid}.json"
    committed = root / "records" / f"{sid}.json"
    if committed.exists():
        return json.loads(committed.read_text())
    if rejection.exists():
        return None

    def reject(reason):
        rejection.parent.mkdir(exist_ok=True)
        atomic_json(rejection, dict(sample_id=sid, reason=reason))
        return None

    start, end = sample["target_time"], sample["target_end"]
    dw = catalog(ee, root / "cache", "dw", sample, start, end)
    dw_ids = [s["asset_id"] for s in dw]
    cover_expression, cover = fetch_cover(ee, sample, config, dw_ids)
    reliable = cover_targets(cover, config["min_known_fraction"])["water_valid"].sum()
    if reliable < config["min_cover_pixels"]:
        return reject("no reliable coverage labels")
    checkpoint = root / "cache" / "inputs" / f"{sid}.npz"
    checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["config_hash"]) != fingerprint(config):
                raise ValueError("Input checkpoint contract changed")
            inputs = {name: saved[name] for name in RetrievalSchema().inputs}
            scenes = json.loads(str(saved["scenes"]))
            eligible = bool(saved["eligible"])
    else:
        inputs, scenes, eligible = input_arrays(ee, sample, root / "cache", norm, config)
        temporary = checkpoint.with_suffix(".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **inputs,
                scenes=json.dumps(scenes),
                eligible=eligible,
                config_hash=fingerprint(config),
            )
        os.replace(temporary, checkpoint)
    if not eligible:
        return reject("insufficient input slot coverage")
    pml, sm = fetch_products(ee, sample, cover_expression, calendar[start])
    targets = build_targets(sample, cover, pml, sm, config)
    if targets["water_valid"].sum() < config["min_cover_pixels"]:
        return reject("no reliable coverage labels")
    validate_arrays(inputs, targets)
    base = root / "zarr" / sid
    # A failed pair commit is entirely owned by this sample and can be rebuilt.
    for name in ("inputs.zarr", "labels.zarr", "inputs.zarr.partial", "labels.zarr.partial"):
        path = base / name
        if path.exists():
            shutil.rmtree(path)
    writer = SampleShardWriter(
        str(base / "inputs.zarr"),
        str(base / "labels.zarr"),
        1,
        schema=RetrievalSchema(),
        attrs=dict(
            sample=sample,
            input_profile=PROFILE,
            scenes=scenes,
            label_sources=dict(dynamic_world=dw_ids, pml=calendar[start]),
            sm_quality=sm,
            config_sha256=fingerprint(config),
            observation_time_semantics="slot_midpoint_reference",
            normalization="S1/S2 pretrained; L1 TOA physical scales",
            normalization_hash=fingerprint(norm),
        ),
    )
    writer.write(inputs, targets)
    writer.finalize()
    row = dict(
        sample,
        input_path=str((base / "inputs.zarr").relative_to(root)),
        label_path=str((base / "labels.zarr").relative_to(root)),
        shard_index=0,
        valid_label_counts={
            name: int(value.sum()) for name, value in targets.items() if name.endswith("_valid")
        },
    )
    committed.parent.mkdir(exist_ok=True)
    atomic_json(committed, row)
    checkpoint.unlink(missing_ok=True)
    return row


def publish(root, rows, config, *, complete):
    from hydrostate.storage.packs import file_hash

    table = pd.DataFrame(
        [
            dict(row, transform=json.dumps(row["transform"]), block=json.dumps(row["block"]))
            for row in rows
        ]
    )
    temporary = root / "samples.parquet.part"
    table.to_parquet(temporary, index=False)
    os.replace(temporary, root / "samples.parquet")
    ledger = SamplingLedger(rows)
    report = dict(
        version=VERSION,
        input_profile=PROFILE,
        complete=complete,
        samples=len(rows),
        requested=config["samples"],
        quotas=quotas(config["samples"]),
        frequency_counts=dict(ledger.counts),
        unique_locations=len(ledger.dates),
        rejected_candidates=sum(1 for _ in (root / "rejected").glob("*.json")),
        valid_label_counts=dict(
            sum((Counter(row.get("valid_label_counts", {})) for row in rows), Counter())
        ),
        year_counts=dict(ledger.years),
        month_counts=dict(ledger.months),
        climate_counts=dict(ledger.climates),
        sm_supervision="tile_mean_weak; 9km grid + approximate 33km quality screen",
        manifest_sha256=file_hash(root / "samples.parquet"),
    )
    atomic_json(root / "collection_report.json", report)
    if complete:
        report["completed_at"] = datetime.now(UTC).isoformat()
        atomic_json(root / "_COLLECTION_SUCCESS.json", report)


def collect(config):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".collection.lock", blocking=False):
        identity = dict(config, version=VERSION, input_profile=PROFILE)
        path = root / "collection.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Collection contract changed; use a new output directory")
        if not path.exists() and any(p.name != ".collection.lock" for p in root.iterdir()):
            raise ValueError("Refusing to adopt an existing dataset")
        atomic_json(path, identity)
        rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
        ledger = SamplingLedger(rows, config["max_dates_per_location"], config["min_date_gap_days"])
        target = quotas(config["samples"])
        if any(ledger.counts[k] > target[k] for k in target):
            raise ValueError("Committed frequency quota exceeded")
        if ledger.counts == Counter(target):
            publish(root, rows, config, complete=True)
            return
        ee, norm = initialize(config["project"]), load_computed_config()
        calendar = pml_calendar(ee, config, root / "cache")
        rng = random.Random(config["seed"])
        dates = sorted(calendar)
        rng.shuffle(dates)
        tried = set()
        last_publish = 0.0
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as workers:
            # Traverse fresh locations before using additional dates. Subsequent
            # passes reuse the on-disk candidate pool and only refill missing bins.
            for _ in range(config["max_dates_per_location"]):
                for candidates in candidate_blocks(ee, config):
                    proposals = deque()
                    for candidate in sorted(candidates, key=lambda s: ledger.priority(s, dates[0])):
                        fb = candidate["frequency_bin"]
                        if ledger.counts[fb] >= target[fb]:
                            continue
                        options = sorted(dates, key=lambda day: ledger.priority(candidate, day))
                        for day in options:
                            identity = (location_key(candidate), day)
                            if identity in tried or not ledger.allows(candidate, day):
                                continue
                            tried.add(identity)
                            proposals.append(dict(candidate, target_time=day))
                            break
                    # Bounded batches, reserving frequency and overlap constraints.
                    while proposals:
                        batch = []
                        reserved = SamplingLedger(
                            (), config["max_dates_per_location"], config["min_date_gap_days"]
                        )
                        while proposals and len(batch) < config["concurrency"]:
                            sample = proposals.popleft()
                            fb = sample["frequency_bin"]
                            if (
                                ledger.counts[fb] + reserved.counts[fb] < target[fb]
                                and ledger.allows(sample, sample["target_time"])
                                and reserved.allows(sample, sample["target_time"])
                            ):
                                batch.append(sample)
                                reserved.add(sample)
                        futures = [
                            workers.submit(collect_sample, ee, s, root, config, norm, calendar)
                            for s in batch
                        ]
                        for future in futures:
                            row = (
                                future.result()
                            )  # Network failures stop; they are never cached as rejection.
                            if row is not None:
                                rows.append(row)
                                ledger.add(row)
                        if rows and time.monotonic() - last_publish >= 60:
                            publish(root, rows, config, complete=False)
                            last_publish = time.monotonic()
                        LOG.info(
                            "Accepted %d/%d; bins=%s",
                            len(rows),
                            config["samples"],
                            dict(ledger.counts),
                        )
                        if all(ledger.counts[k] == target[k] for k in target):
                            publish(root, rows, config, complete=True)
                            return
        raise RuntimeError(
            "Candidate pool exhausted before hard quotas; inspect collection_report.json"
        )
