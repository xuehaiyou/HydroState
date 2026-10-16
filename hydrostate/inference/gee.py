"""Input-only Earth Engine retrieval with a reusable, spatially chunked scene cache."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import shutil
import sqlite3
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform

from hydrostate.data.gee_sampling import (
    BANDS,
    MODALITIES,
    atomic_json,
    collection,
    fingerprint,
    prepare,
    slots,
)
from hydrostate.inference.planning import (
    input_shape,
    input_transform,
    map_point,
    overlap_records,
    window_geometry,
    windows,
)
from hydrostate.storage.schema import Schema

LOG = logging.getLogger(__name__)


@contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle, flags)
        yield


def retry(function, attempts=5):
    for attempt in range(attempts):
        try:
            return function()
        except Exception:
            if attempt == attempts - 1:
                raise
            LOG.warning("GEE request failed; retry %d/%d", attempt + 1, attempts)
            time.sleep(min(2**attempt, 16))


def normalize_pixels(key, values, valid, norm):
    """Exactly the training ingestion normalization, including float16 rounding."""
    output = np.zeros(values.shape, dtype=np.float16)
    for index, band in enumerate(BANDS[key]):
        canonical = (
            band.lower()
            if key == "s1"
            else ("B" + band[1:].zfill(2) if key == "s2" and band != "B8A" else band)
        )
        stats = norm[MODALITIES[key]][canonical]
        value = (values[index].astype(np.float32) - stats["mean"]) / (4 * stats["std"]) + 0.5
        output[index] = np.where(valid, value, 0)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite normalized input")
    return output


class SceneCache:
    """One ordinary Zarr group per scene; only requested spatial chunks exist.

    A scene lock serializes its writes. The done bitmap is written last, so an
    interrupted download is safely repeated. Inference is strictly read-only.
    """

    def __init__(self, root, plan, *, ee=None, min_free_gib=5):
        self.root, self.plan, self.ee = Path(root), plan, ee
        if not math.isfinite(min_free_gib) or min_free_gib < 5:
            raise ValueError("min_free_gib must be finite and at least 5")
        self.min_free_gib = min_free_gib
        self.stopped = Event()
        self.grid, self.chunk = plan["grid"], plan["config"]["cache_chunk"]
        if not 120 <= self.chunk <= 720 or self.chunk % 3:
            raise ValueError("cache_chunk must be a multiple of 3 in [120,720]")
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
            load_computed_config,
        )

        self.norm = load_computed_config()
        self._groups = OrderedDict()
        self.contract = fingerprint(
            dict(
                grid=self.grid,
                normalization=self.norm,
                bands=BANDS,
                preprocessing="gee_sampling.prepare_v1",
            )
        )

    def path(self, scene):
        return self.root / "input_cache" / scene["key"] / (fingerprint(scene["asset_id"]) + ".zarr")

    def _open(self, scene, writable=False):
        path = self.path(scene)
        if not writable:
            if str(path) in self._groups:
                self._groups.move_to_end(str(path))
                return self._groups[str(path)]
            group = zarr.open_group(str(path), mode="r")
        else:
            group = zarr.open_group(str(path), mode="a")
            if "contract" not in group.attrs:
                h, w = input_shape(self.grid)
                bands, chunk = len(BANDS[scene["key"]]), self.chunk
                compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
                group.require_dataset(
                    "values",
                    shape=(bands, h, w),
                    chunks=(bands, chunk, chunk),
                    dtype="f2",
                    compressor=compressor,
                )
                group.require_dataset(
                    "valid", shape=(h, w), chunks=(chunk, chunk), dtype="u1", compressor=compressor
                )
                done_shape = (math.ceil(h / chunk), math.ceil(w / chunk))
                group.require_dataset(
                    "done", shape=done_shape, chunks=done_shape, dtype="u1", compressor=compressor
                )
                group.attrs.update(contract=self.contract, scene=scene)
        if group.attrs["contract"] != self.contract:
            raise ValueError("Scene cache preprocessing/grid mismatch")
        if not writable:
            self._groups[str(path)] = group
            if len(self._groups) > 64:
                self._groups.popitem(last=False)
        return group

    def _download(self, scene, cy, cx, group):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        if self.ee is None:
            raise FileNotFoundError(f"Scene chunk is not prepared: {scene['asset_id']} {cy},{cx}")
        if shutil.disk_usage(self.root).free < self.min_free_gib * 1024**3:
            raise OSError(
                f"Less than {self.min_free_gib:g} GiB free; free space and rerun to resume"
            )
        h, w = input_shape(self.grid)
        y, x = cy * self.chunk, cx * self.chunk
        height, width = min(self.chunk, h - y), min(self.chunk, w - x)
        ee, key = self.ee, scene["key"]
        image = prepare(ee, key, ee.Image(scene["asset_id"]))
        valid = image.mask().reduce(ee.Reducer.min()).gt(0)
        expression = ee.Image.cat(
            [
                image.unmask(0, sameFootprint=False),
                valid.unmask(0, sameFootprint=False).rename("valid"),
            ]
        ).toFloat()
        tr = input_transform(self.grid)
        left, top = map_point(tr, x, y)
        names = [*BANDS[key], "valid"]
        payload = retry(
            lambda: ee.data.computePixels(
                dict(
                    expression=expression,
                    fileFormat="NUMPY_NDARRAY",
                    bandIds=names,
                    grid=dict(
                        dimensions=dict(width=width, height=height),
                        crsCode=self.grid["crs"],
                        affineTransform=dict(
                            scaleX=10,
                            shearX=0,
                            translateX=left,
                            shearY=0,
                            scaleY=-10,
                            translateY=top,
                        ),
                    ),
                )
            )
        )
        if payload.shape != (height, width) or list(payload.dtype.names or ()) != names:
            raise ValueError("Unexpected computePixels shape/band order")
        values = np.stack([payload[b] for b in BANDS[key]])
        valid = payload["valid"] > 0
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite GEE response")
        group["values"][:, y : y + height, x : x + width] = normalize_pixels(
            key, values, valid, self.norm
        )
        group["valid"][y : y + height, x : x + width] = valid
        group["done"][cy, cx] = 1

    def read(self, scene, window):
        if self.stopped.is_set():
            raise RuntimeError("Input preparation stopped after a worker failure")
        y, x = window["row"] * 3, window["col"] * 3
        indices = [
            (cy, cx)
            for cy in range(y // self.chunk, (y + 119) // self.chunk + 1)
            for cx in range(x // self.chunk, (x + 119) // self.chunk + 1)
        ]
        path = self.path(scene)
        if self.ee is not None:
            with locked(str(path) + ".lock"):
                group = self._open(scene, writable=True)
                for cy, cx in indices:
                    if not group["done"][cy, cx]:
                        self._download(scene, cy, cx, group)
        else:
            group = self._open(scene)
            if any(not group["done"][cy, cx] for cy, cx in indices):
                raise FileNotFoundError(f"Incomplete cached scene: {scene['asset_id']}")
        return group["values"][:, y : y + 120, x : x + 120], group["valid"][
            y : y + 120, x : x + 120
        ].astype(bool)


def catalog(root, plan, key, year, ee):
    dest = Path(root) / "catalogs" / f"{key}_{year}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid = plan["grid"]
    h, w = input_shape(grid)
    tr = input_transform(grid)
    area = ee.Geometry.Rectangle([tr.c, tr.f - h * 10, tr.c + w * 10, tr.f], grid["crs"], False)
    images = collection(ee, key).filterBounds(area).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    properties = [
        "asset_id",
        "system:time_start",
        "relativeOrbitNumber_start",
        "orbitProperties_pass",
    ]
    features = ee.FeatureCollection(
        images.toList(images.size()).map(
            lambda im: ee.Feature(
                ee.Image(im).geometry().bounds(100), ee.Image(im).toDictionary(properties)
            )
        )
    )
    data = retry(features.getInfo)
    project = Transformer.from_crs(4326, grid["crs"], always_xy=True).transform
    scenes = []
    for feature in data["features"]:
        props = feature["properties"]
        scenes.append(
            dict(
                key=key,
                asset_id=props["asset_id"],
                time=props["system:time_start"] // 1000,
                orbit=props.get("relativeOrbitNumber_start"),
                bounds=list(transform(project, shape(feature["geometry"])).bounds),
            )
        )
    scenes.sort(key=lambda x: (x["time"], x["asset_id"]), reverse=True)
    atomic_json(dest, scenes)
    LOG.info("Catalog %s %s: %d scenes", key, year, len(scenes))
    return scenes


def epoch(day):
    return int(datetime.fromisoformat(str(day)).replace(tzinfo=UTC).timestamp())


def select_window(plan, period, window, scenes, cache):
    cfg, selected, orbit = plan["config"], {}, None
    footprint = window_geometry(plan["grid"], window)
    for slot, (start, end) in reversed(list(enumerate(slots(period["time"])))):
        a, b = epoch(start), epoch(end)
        for key in BANDS:
            choice = None
            for scene in scenes[key]:
                if not a <= scene["time"] < b:
                    continue
                if key == "s1" and orbit is not None and scene["orbit"] != orbit:
                    continue
                if not box(*scene["bounds"]).intersects(footprint):
                    continue
                _, valid = cache.read(scene, window)
                # Evaluate the 30 m pixel centres on our explicitly aligned grid.
                fraction = float(valid[1::3, 1::3].mean())
                if fraction >= cfg["min_valid_fraction"]:
                    choice = dict(scene, valid_fraction=fraction)
                    break
            selected[f"{key}_{slot}"] = choice
            if key == "s1" and choice:
                orbit = choice["orbit"]
    active = {name.rsplit("_", 1)[1] for name, item in selected.items() if item}
    latest = max((s["time"] for s in selected.values() if s), default=0)
    # Match the training requirement at day resolution (acquisitions on t are allowed).
    age = max(0, (epoch(period["time"]) - latest) / 86400) if latest else None
    accepted = (
        len(active) >= cfg["min_active_slots"] and age is not None and age <= cfg["latest_days"]
    )
    return dict(
        window,
        time=period["time"],
        accepted=accepted,
        active_slots=len(active),
        latest_age_days=age,
        selections=json.dumps(selected, sort_keys=True),
        **{
            k: json.dumps(v)
            for k, v in overlap_records(plan["grid"], window, period, plan["training"]).items()
        },
    )


def prepare_date(root, plan, period, *, workers=4, min_free_gib=5):
    import ee

    root = Path(root)
    ee.Initialize(project=plan["config"]["gee_project"])
    ee.data.setDeadline(180000)
    dest = root / "manifests" / f"{period['time']}.parquet"
    if dest.exists():
        table = pd.read_parquet(dest)
        if len(table) != len(windows(plan)):
            raise ValueError("Incomplete date manifest")
        return table
    with locked(root / "prepare.lock", blocking=False):
        t = date.fromisoformat(period["time"])
        years = range((t - timedelta(days=63)).year, t.year + 1)
        scenes = {
            key: sorted(
                [s for year in years for s in catalog(root, plan, key, year, ee)],
                key=lambda s: (s["time"], s["asset_id"]),
                reverse=True,
            )
            for key in BANDS
        }
        cache = SceneCache(root, plan, ee=ee, min_free_gib=min_free_gib)
        todo = windows(plan)
        db = sqlite3.connect(root / "preparation.sqlite")
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS windows (date TEXT, id TEXT, result TEXT, "
                "PRIMARY KEY(date,id))"
            )
            saved = {
                wid: json.loads(value)
                for wid, value in db.execute(
                    "SELECT id,result FROM windows WHERE date=?", (period["time"],)
                )
            }
            pending = [w for w in todo if w["window_id"] not in saved]
            accepted_count = sum(r["accepted"] for r in saved.values())
            LOG.info(
                "%s resuming %d/%d windows; free=%.1f GiB, reserve=%.1f GiB",
                period["time"],
                len(saved),
                len(todo),
                shutil.disk_usage(root).free / 1024**3,
                min_free_gib,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                remaining = iter(pending)
                futures = set()

                def submit_next():
                    window = next(remaining, None)
                    if window is not None:
                        futures.add(pool.submit(select_window, plan, period, window, scenes, cache))

                for _ in range(workers):
                    submit_next()
                try:
                    while futures:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            result = future.result()
                            saved[result["window_id"]] = result
                            accepted_count += result["accepted"]
                            db.execute(
                                "INSERT INTO windows VALUES (?,?,?)",
                                (period["time"], result["window_id"], json.dumps(result)),
                            )
                            db.commit()
                            if (
                                len(saved) == len(todo)
                                or len(saved) % max(1, len(todo) // 100) == 0
                            ):
                                LOG.info(
                                    "%s prepared %d/%d windows; accepted=%d",
                                    period["time"],
                                    len(saved),
                                    len(todo),
                                    accepted_count,
                                )
                        for _ in completed:
                            submit_next()
                except BaseException:
                    cache.stopped.set()
                    for future in futures:
                        future.cancel()
                    raise
            table = pd.DataFrame([saved[w["window_id"]] for w in todo])
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".parquet.part")
            table.to_parquet(temporary, index=False)
            os.replace(temporary, dest)
            return table
        finally:
            db.close()


def assemble_inputs(cache, row, *, drop_s2=False):
    values = {name: np.zeros(spec.tail_shape, spec.dtype) for name, spec in Schema().inputs.items()}
    selected = json.loads(row["selections"])
    for slot in range(4):
        for mi, key in enumerate(BANDS):
            scene = selected[f"{key}_{slot}"]
            if scene is None or (drop_s2 and key == "s2"):
                continue
            image, valid = cache.read(scene, row)
            values[key][slot] = image
            values[f"{key}_valid"][slot, 0] = valid
            values["sensor_valid"][slot, mi] = bool(valid.any())
            values["observation_time"][slot, mi] = scene["time"]
    return values
