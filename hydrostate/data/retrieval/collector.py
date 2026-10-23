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

from .alignment import grid_metadata, sensor_masks
from .catalog import candidate_blocks, catalog, pml_calendar
from .composites import QUALITY, normalize, sensor_composite
from .contract import INPUT_GROUPS, PROFILE, VERSION, RetrievalSchema, intervals, product_end
from .gee_labels import fetch_cover, fetch_products
from .labels import build_targets, cover_targets, sm_tile_eligible
from .sampling import SamplingLedger, location_key, quotas
from .scheduling import completed_samples
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
    for name, group in INPUT_GROUPS.items():
        sensor = group.sensor
        chunks, packed_names, present = [], [], []
        names = [*group.bands, *QUALITY]
        for slot, (start, end) in enumerate(slots):
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
            expression = sensor_composite(ee, sensor, ids, start, group.bands)
            renamed = [f"t{slot}_{name}" for name in names]
            chunks.append(expression.rename(renamed))
            packed_names.extend(renamed)
            present.append(slot)
        if not chunks:
            continue
        # One bounded request per resolution group, retaining all five slots.
        height, width = group.shape(schema.height, schema.width)
        packed = compute_pixels(
            ee,
            ee.Image.cat(chunks),
            packed_names,
            sample["crs"],
            group.transform(sample["transform"]),
            width,
            height,
        )
        for slot in present:
            start, end = slots[slot]
            raw = {band: packed[f"t{slot}_{band}"] for band in names}
            valid = raw["days"] > 0
            arrays[name][slot] = normalize(
                sensor, np.stack([raw[band] for band in group.bands]), valid, norm, group.bands
            )
            arrays[f"{name}_valid"][slot, 0] = valid
            arrays[f"{name}_quality"][slot] = np.stack([raw[band] for band in QUALITY])
    masks = sensor_masks(arrays, (schema.height, schema.width))
    for modality, sensor in enumerate(BANDS):
        available = masks[sensor].any(axis=(1, 2, 3))
        arrays["sensor_valid"][:, modality] = available
        # Structural references only; actual dates stay in per-group quality.
        arrays["observation_time"][:, modality] = np.where(
            available, arrays["slot_bounds"].mean(axis=1).astype(np.int64), 0
        )
    active = sum(
        any(masks[sensor][slot].mean() >= config["min_input_fraction"] for sensor in BANDS)
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
    coverage = cover_targets(cover, config["min_known_fraction"])
    reliable = coverage["water_valid"].sum()
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
    pml, sm = fetch_products(
        ee,
        sample,
        cover_expression,
        calendar[start],
        config,
        root / "cache",
        sm_enabled=sm_tile_eligible(coverage, config),
    )
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
            input_grids=grid_metadata(sample),
            local_alignment="pixel_center_nearest; common_tile_extent; model_10m",
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

                    def collect_one(sample):
                        return collect_sample(ee, sample, root, config, norm, calendar)

                    for row in completed_samples(
                        workers, proposals, ledger, target, config["concurrency"], collect_one
                    ):
                        if row is not None:
                            rows.append(row)
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
