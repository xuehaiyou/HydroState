"""Resumable bounded collection: centre proposals -> validated immutable shards."""

import json
import logging
import os
import random
import shutil
import time
from collections import Counter
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
from .candidates import WaterCandidateSource
from .catalog import catalog, pml_calendar
from .composites import QUALITY, normalize, radar_orbit, sensor_composite
from .contract import INPUT_GROUPS, PROFILE, VERSION, RetrievalSchema, intervals, product_end
from .gee_labels import fetch_cover, fetch_products
from .ice_candidates import IceCandidateSource
from .labels import build_targets, cover_targets, ice_centre_eligible, sm_tile_eligible
from .plan import bind_plan, collection_stop, snapshot_release
from .sampling import ICE_GROUP, SAMPLING_VERSION, SamplingLedger, location_key, quotas
from .scheduling import completed_samples
from .specs import specification_snapshot
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
    orbit = radar_orbit(catalogs["s1"])
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
        atomic_json(rejection, dict(sample, reason=reason))
        return None

    start, end = sample["target_time"], sample["target_end"]
    dw = catalog(ee, root / "cache", "dw", sample, start, end)
    dw_ids = [s["asset_id"] for s in dw]
    cover_expression, cover = fetch_cover(ee, sample, config, dw_ids)
    coverage = cover_targets(cover, config["min_known_fraction"])
    reliable = coverage["water_valid"].sum()
    if reliable < config["min_cover_pixels"]:
        return reject("no reliable coverage labels")
    if sample["sampling_group"] == ICE_GROUP and not ice_centre_eligible(
        coverage, config["min_center_ice_fraction"]
    ):
        return reject("insufficient target-window centre ice")
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
            specifications=specification_snapshot(),
        ),
    )
    writer.write(inputs, targets)
    writer.finalize()
    row = dict(
        sample,
        max_snow_ice=float(np.where(coverage["snow_ice_valid"], coverage["snow_ice"], 0).max()),
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


def publish(root, rows, config, *, complete, ice_search=None):
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
        sampling_version=SAMPLING_VERSION,
        sampling_group_counts=dict(ledger.counts),
        frequency_counts=dict(Counter(row["frequency_bin"] for row in rows)),
        ice_search=dict(ice_search or {}),
        unique_locations=len(ledger.dates),
        rejected_candidates=sum(1 for _ in (root / "rejected").glob("*.json")),
        valid_label_counts=dict(
            sum((Counter(row.get("valid_label_counts", {})) for row in rows), Counter())
        ),
        year_counts=dict(ledger.years),
        month_counts=dict(ledger.months),
        year_month_counts=dict(Counter(row["target_time"][:7] for row in rows)),
        climate_counts=dict(ledger.climates),
        split_counts=dict(Counter(row["split"] for row in rows)),
        geographic_block_counts=dict(Counter(str(row["block"]) for row in rows)),
        valid_label_sample_counts=dict(
            sum(
                (
                    Counter({k: int(v > 0) for k, v in row.get("valid_label_counts", {}).items()})
                    for row in rows
                ),
                Counter(),
            )
        ),
        sm_supervision="tile_mean_weak; 9km grid + approximate 33km quality screen",
        manifest_sha256=file_hash(root / "samples.parquet"),
    )
    atomic_json(root / "collection_report.json", report)
    if complete:
        report["completed_at"] = datetime.now(UTC).isoformat()
        atomic_json(root / "_COLLECTION_SUCCESS.json", report)


def collect(config, *, adopt_from=None):
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config

    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".collection.lock", blocking=False):
        sample_config = bind_plan(root, config, adopt_from=adopt_from)
        state_path = root / "sampling_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
        ledger = SamplingLedger(rows, config["max_dates_per_location"], config["min_date_gap_days"])
        target = quotas(config["samples"])
        if any(ledger.counts[k] > target[k] for k in target):
            raise ValueError("Committed sampling-group quota exceeded")
        if ledger.counts == Counter(target):
            # A completed collection is immutable, including its timestamp and
            # query statistics. Rebuild only a missing final publication.
            if not (root / "_COLLECTION_SUCCESS.json").exists():
                report_path = root / "collection_report.json"
                previous = json.loads(report_path.read_text()) if report_path.exists() else {}
                publish(root, rows, config, complete=True, ice_search=previous.get("ice_search"))
            return
        ee, norm = initialize(config["project"]), load_computed_config()
        calendar = pml_calendar(ee, config, root / "cache")
        rng = random.Random(config["seed"])
        tried = {
            (location_key(row), row["target_time"])
            for path in (root / "rejected").glob("*.json")
            if "target_time" in (row := json.loads(path.read_text()))
        }
        ice_source = IceCandidateSource(ee, config, calendar, rows, state=state.get("ice"))
        water_source = WaterCandidateSource(ee, config, state=state.get("water"))
        last_publish = 0.0
        with (
            collection_stop() as stop,
            ThreadPoolExecutor(max_workers=config["concurrency"]) as workers,
        ):

            def collect_one(sample):
                return collect_sample(ee, sample, root, sample_config, norm, calendar)

            for row in completed_samples(
                workers,
                [],
                ledger,
                target,
                config["concurrency"],
                collect_one,
                dates=calendar,
                rng=rng,
                tried=tried,
                ice_source=ice_source,
                water_source=water_source,
                capacity=config["candidate_pool_capacity"],
                low_watermark=config["candidate_low_watermark"],
                ice_buffer=config["ice_candidate_buffer"],
                ice_low_watermark=config["ice_candidate_low_watermark"],
                state=state,
                checkpoint=lambda value: atomic_json(state_path, value),
                stop=stop,
            ):
                if row is not None:
                    rows.append(row)
                if rows and time.monotonic() - last_publish >= 60:
                    publish(root, rows, config, complete=False, ice_search=ice_source.stats)
                    last_publish = time.monotonic()
                LOG.info(
                    "Accepted %d/%d; sampling_groups=%s",
                    len(rows),
                    config["samples"],
                    dict(ledger.counts),
                )
        complete = all(ledger.counts[k] == target[k] for k in target)
        if rows:
            publish(root, rows, config, complete=complete, ice_search=ice_source.stats)
        if complete:
            snapshot_release(root)
            return
        if stop.is_set():
            LOG.info("Collection stopped cleanly; queues and cursors saved")
            return
        raise RuntimeError(
            "Candidate pool exhausted before hard quotas; inspect collection_report.json"
        )
