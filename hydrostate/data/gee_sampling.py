"""Global GEE sampling and verified Drive transfer. Run with --help.

Public catalogs define band contracts; private climate metadata is checked at runtime.
All network operations are explicit in main(); pure helpers can be tested offline.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import calendar
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import time
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datetime import date, datetime, timedelta, timezone

LOG = logging.getLogger(__name__)
SIZE = 120
SAMPLING_STRATEGY = "climate_month_frequency7_v1"
FREQUENCY_BINS = tuple(range(7))
BANDS = {
    "s1": ["VV", "VH"],
    "s2": ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"],
    "landsat": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
}
COLLECTIONS = {
    "s1": "COPERNICUS/S1_GRD", "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "dw": "GOOGLE/DYNAMICWORLD/V1",
    "pml": "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",
}
MODALITIES = {"s1": "sentinel1", "s2": "sentinel2_l2a", "landsat": "landsat"}


def slots(target):
    t = date.fromisoformat(target)
    return [((t + timedelta(days=a)).isoformat(), (t + timedelta(days=b)).isoformat())
            for a, b in [(-63, -47), (-47, -31), (-31, -15), (-15, 1)]]


def frequency_bin(value):
    if value is None or not math.isfinite(value) or not 0 <= value <= 100:
        return None
    if value == 0:
        return 0
    if value == 100:
        return 6
    return min(5, math.ceil(value / 20))


def stratum_counts(histogram):
    """Normalize GEE histogram keys, excluding masked/null and invalid strata."""
    counts = {}
    for key, value in (histogram or {}).items():
        try:
            code, population = float(key), float(value)
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(code) or not code.is_integer()
                or code < 10 or int(code) % 10 not in FREQUENCY_BINS
                or not math.isfinite(population) or population <= 0):
            continue
        code = int(code)
        counts[code] = counts.get(code, 0) + population
    return counts


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".part")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(tmp, path)


def quota(total, keys):
    keys = sorted(keys)
    return {key: total // len(keys) + (i < total % len(keys)) for i, key in enumerate(keys)}


def climate_quota(total, counts):
    """Half equal allocation, half proportional to eligible area."""
    if not counts:
        return {}
    keys = sorted(counts)
    weights = {k: 0.5 / len(keys) + 0.5 * counts[k] / sum(counts.values()) for k in keys}
    exact = {k: total * weights[k] for k in keys}
    out = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (exact[k] - out[k], counts[k]), reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def grid(lon, lat):
    from pyproj import Transformer
    zone = min(60, int((lon + 180) // 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    if lat >= 84:
        epsg = 3413
    elif lat < -80:
        epsg = 3031
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    left, top = round((x - 600) / 30) * 30, round((y + 600) / 30) * 30
    return f"EPSG:{epsg}", [10, 0, left, 0, -10, top]


def region(ee, sample):
    tr = sample["transform"]
    return ee.Geometry.Rectangle([tr[2], tr[5] - 1200, tr[2] + 1200, tr[5]],
                                 sample["crs"], False)


def prepare(ee, key, image):
    if key == "s1":
        return image.select(BANDS[key])
    if key == "s2":
        scl = image.select("SCL")
        valid = scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
        for code in [8, 9, 10]:
            valid = valid.And(scl.neq(code))
        return image.select(BANDS[key]).updateMask(valid)
    if key == "landsat":
        valid = image.select("QA_PIXEL").bitwiseAnd(63).eq(0)
        valid = valid.And(image.select("QA_RADSAT").eq(0))
        return image.select(BANDS[key]).updateMask(valid)
    if key == "dw":
        return image.select("label").eq(0).rename("water")
    if key == "sm":
        sm = image.select("soil_moisture_am")
        return (sm.rename("soil_moisture")
                .updateMask(image.select("retrieval_qual_flag_am").eq(0))
                .updateMask(sm.gte(0).And(sm.lte(1))))
    et = image.select("ET").multiply(0.01)
    return et.rename("et").updateMask(et.gte(0))


def collection(ee, key):
    if key == "landsat":
        return ee.ImageCollection("LANDSAT/LC08/C02/T1").map(
            lambda im: im.set("asset_id", ee.String("LANDSAT/LC08/C02/T1/").cat(im.get("system:index")))).merge(
            ee.ImageCollection("LANDSAT/LC09/C02/T1").map(
                lambda im: im.set("asset_id", ee.String("LANDSAT/LC09/C02/T1/").cat(im.get("system:index")))))
    if key == "sm":
        return (ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
                .filterDate("2015-01-01", "2023-12-04").map(lambda im: im.set("asset_id", ee.String("NASA/SMAP/SPL3SMP_E/005/").cat(im.get("system:index")))).merge(
                    ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
                    .filterDate("2023-12-04", "2100-01-01").map(lambda im: im.set("asset_id", ee.String("NASA/SMAP/SPL3SMP_E/006/").cat(im.get("system:index"))))))
    c = ee.ImageCollection(COLLECTIONS[key])
    if key == "s1":
        c = (c.filter(ee.Filter.eq("instrumentMode", "IW"))
             .filter(ee.Filter.eq("resolution_meters", 10))
             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return c.map(lambda im: im.set("asset_id", ee.String(COLLECTIONS[key] + "/").cat(im.get("system:index"))))


def select_scene(ee, key, sample, start, end, target, minimum, orbit=None):
    c = collection(ee, key).filterBounds(region(ee, sample)).filterDate(start, end)
    if orbit is not None:
        c = c.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
    target_ms = ee.Date(target).millis()

    def score(im):
        valid = prepare(ee, key, im).mask().reduce(ee.Reducer.min()).unmask(0)
        fraction = valid.reduceRegion(ee.Reducer.mean(), region(ee, sample),
                                      scale=30, maxPixels=20000).values().get(0)
        return im.set({"valid_fraction": fraction,
                       "distance": ee.Number(im.get("system:time_start")).subtract(target_ms).abs()})

    c = c.map(score).filter(ee.Filter.gte("valid_fraction", minimum)).sort("distance")
    # A FeatureCollection returns metadata only, never image pixels through getInfo.
    result = ee.FeatureCollection(c.limit(1).toList(1).map(
        lambda im: ee.Feature(None, ee.Image(im).toDictionary(
            ["system:time_start", "system:index", "valid_fraction", "relativeOrbitNumber_start",
             "orbitProperties_pass", "MGRS_TILE", "system:time_end"]))
        .set("asset_id", ee.Image(im).get("asset_id")))).getInfo()["features"]
    return result[0]["properties"] if result else None


def screen_block(ee, kg, area):
    """Conservative land screen at the climate raster's native resolution.

    Buffer the boundary to retain climate pixels overlapping either side of a
    block edge. No JRC 30 m computation is requested for an empty climate block.
    """
    eligible = kg.gt(0).rename("eligible")
    result = eligible.reduceRegion(
        reducer=ee.Reducer.max().unweighted(), geometry=area.buffer(2000),
        crs=kg.projection(), scale=kg.projection().nominalScale(),
        maxPixels=10000000, tileScale=4).getInfo()
    return bool(result.get("eligible"))


def candidate_pool(ee, args, root, kg, strata, sampling_crs):
    """Shared spatial cache; serialize builders, then release before date selection."""
    import fcntl
    cache = args.candidate_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    expected = dict(sampling_strategy=SAMPLING_STRATEGY, climate_asset=args.climate_asset,
                    climate_band=args.climate_band, candidate_seed=args.candidate_seed,
                    candidates_per_class=args.candidates_per_class)
    with ExitStack() as stack:
        # Existing pre-sharing producers lock their run directory, not the pool.
        # A symlink to their candidates preserves the running writer's paths.
        source = cache.parent
        if (source != root and (source / "config.json").exists() and cache.name == "candidates"
                and not all((cache / f"{i:04}.json").exists() for i in range(12960))):
            old_lock = stack.enter_context((source / ".lock").open("a"))
            try:
                fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The source sampling run still owns this candidate pool; "
                                   "let it finish before starting another run") from None
        lock = stack.enter_context((cache / ".lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Candidate pool is being built/read by another run; retry later") from None
        check_candidate_config(cache, expected)
        blocks = [(x, y) for y in range(-60, 84, 2) for x in range(-180, 180, 2)]
        candidates, counts = [], {}
        rng = random.Random(args.seed)
        for index, (x, y) in enumerate(blocks):
            dest = cache / f"{index:04}.json"
            if not dest.exists():
                area = ee.Geometry.Rectangle([x, y, x + 2, min(y + 2, 84)], geodesic=False)
                if not screen_block(ee, kg, area):
                    atomic_json(dest, {"features": [], "counts": {}, "screened_out": True})
                else:
                    features = strata.stratifiedSample(
                        numPoints=args.candidates_per_class, classBand="stratum", region=area,
                        projection=sampling_crs, scale=30, seed=args.candidate_seed + index,
                        tileScale=4, geometries=True).getInfo()["features"]
                    histogram = strata.reduceRegion(ee.Reducer.frequencyHistogram().unweighted(),
                        area, scale=30, crs=sampling_crs, maxPixels=3000000000, tileScale=4).getInfo().get("stratum", {})
                    atomic_json(dest, {"features": features, "counts": histogram})
            cached = json.loads(dest.read_text())
            if cached.get("screened_out"):
                LOG.info("Candidate block %s/%s; skipped=no climate coverage; pool=%s",
                         index + 1, len(blocks), len(candidates))
                continue
            histogram = stratum_counts(cached.get("counts"))
            local_counts = {}
            for f in cached["features"]:
                code = int(f["properties"]["stratum"])
                local_counts[code] = local_counts.get(code, 0) + 1
            for code, n in histogram.items():
                counts[code] = counts.get(code, 0) + n
            for f in cached["features"]:
                code = int(f["properties"]["stratum"])
                population = histogram.get(code, 1)
                f["priority"] = -math.log(max(rng.random(), 1e-12)) / (population / local_counts[code])
                candidates.append(f)
            LOG.info("Candidate block %s/%s; pool=%s", index + 1, len(blocks), len(candidates))
    return candidates, counts


def check_candidate_config(cache, expected):
    """Adopt a known existing seven-bin pool, or enforce its immutable contract."""
    path = cache / "pool_config.json"
    if path.exists():
        actual = json.loads(path.read_text())
    elif any(cache.glob("[0-9][0-9][0-9][0-9].json")):
        source = cache.parent / "config.json"
        if not source.exists():
            raise RuntimeError("Existing candidate pool has no configuration metadata")
        run = json.loads(source.read_text())
        actual = {k: run.get(k) for k in expected}
        actual["candidate_seed"] = run.get("candidate_seed", run.get("seed"))
    else:
        actual = expected
    if actual != expected:
        raise RuntimeError("Candidate pool configuration differs; use another CANDIDATE_DIR")
    if not path.exists():
        atomic_json(path, expected)


def plan(ee, args, root):
    """Cache spatial blocks, then fill exact frequency/month quotas from that pool."""
    import pandas as pd
    path = root / "sample_index.parquet"
    if path.exists():
        return pd.read_parquet(path).to_dict("records")
    from pyproj import CRS
    sampling_crs = CRS.from_epsg(6933).to_wkt(version="WKT1_GDAL")
    climate = ee.Image(args.climate_asset)
    info = climate.getInfo()
    band = args.climate_band or info["bands"][0]["id"]
    if band not in [b["id"] for b in info["bands"]]:
        raise ValueError(f"Climate band {band!r} absent")
    atomic_json(root / "climate_metadata.json", info)
    raw_climate = climate.select(band)
    kg = raw_climate.toInt()
    kg = kg.updateMask(raw_climate.eq(kg))
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    # unmask(0) removes fractional occurrence masks; build a new Boolean mask.
    occ = occ.mask(ee.Image.constant(1))
    # Yearly class 0 is no-data; classes 1/2/3 establish observed land/water.
    # This prevents unmask(0) from treating unobserved pixels as dry land.
    observed = (ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
                .select("waterClass").max().gt(0))
    freq = occ.subtract(1).divide(20).floor().add(1).toInt().where(occ.eq(100), 6)
    strata = kg.multiply(10).add(freq).rename("stratum")
    strata = strata.updateMask(occ.gte(0).And(occ.lte(100)).And(kg.gt(0)).And(observed))
    candidates, counts = candidate_pool(ee, args, root, kg, strata, sampling_crs)
    rng = random.Random(args.seed)
    candidates.sort(key=lambda f: f["priority"])
    pools = {}
    for f in candidates:
        code = int(f["properties"]["stratum"])
        pools.setdefault(code, []).append(f["geometry"]["coordinates"])
    years = range(args.start_year, args.end_year + 1)
    months = [f"{y}-{m:02}" for y in years for m in range(1, 13)]
    rows, used = [], set()
    location_counts, occupied = {}, {}
    selection_cache = root / "selections"
    selection_cache.mkdir(exist_ok=True)
    for fb, count in quota(args.samples, FREQUENCY_BINS).items():
        classes = sorted(k for k in pools if k % 10 == fb)
        if not classes and count:
            raise RuntimeError(f"No candidates for frequency bin {fb}")
        for month, n in quota(count, months).items():
            for code, required in climate_quota(n, {k: counts[k] for k in classes}).items():
                accepted = 0
                for lon, lat in pools[code]:
                    if accepted == required:
                        break
                    crs, transform = grid(lon, lat)
                    identity = (month, crs, transform[2], transform[5])
                    if identity in used:
                        continue
                    used.add(identity)
                    location = (crs, transform[2], transform[5])
                    if location_counts.get(location, 0) >= args.max_dates_per_location:
                        continue
                    occupied_key = (month, crs)
                    if any(abs(transform[2] - x) < 1200 and abs(transform[5] - y) < 1200
                           for x, y in occupied.get(occupied_key, [])):
                        continue
                    year, m = map(int, month.split("-"))
                    day = rng.randint(1, calendar.monthrange(year, m)[1])
                    target = date(year, m, day)
                    start = date(year, m, 1)
                    end = start + timedelta(days=calendar.monthrange(year, m)[1])
                    # Geographic split blocks, fixed across dates. Exclude 12-km
                    # border buffers to reduce shared coarse-label leakage.
                    dx = min((lon + 180) % 5, 5 - (lon + 180) % 5) * 111320 * math.cos(math.radians(lat))
                    dy = min((lat + 90) % 5, 5 - (lat + 90) % 5) * 111320
                    if min(dx, dy) < 12000:
                        continue
                    block = f"{int((lon + 180) // 5)}:{int((lat + 90) // 5)}"
                    h = int(fingerprint([args.seed, block])[:8], 16) % 100
                    s = dict(lon=lon, lat=lat, crs=crs, transform=transform,
                             climate_class=code // 10, frequency_bin=fb,
                             split="train" if h < 70 else "validation" if h < 85 else "test")
                    cache_file = selection_cache / (fingerprint(identity) + ".json")
                    if cache_file.exists():
                        saved = json.loads(cache_file.read_text())
                        if saved:
                            rows.append(saved)
                            location_counts[location] = location_counts.get(location, 0) + 1
                            occupied.setdefault(occupied_key, []).append((transform[2], transform[5]))
                            accepted += 1
                        continue
                    dw = select_scene(ee, "dw", s, str(start), str(end), str(target), args.min_valid)
                    if not dw:
                        atomic_json(cache_file, None)
                        continue
                    t = datetime.fromtimestamp(dw["system:time_start"] / 1000, timezone.utc).date()
                    inputs, orbit = {}, None
                    for slot, (a, b) in reversed(list(enumerate(slots(str(t))))):
                        for key in BANDS:
                            selected = select_scene(ee, key, s, a, b, str(t), args.min_valid, orbit if key == "s1" else None)
                            inputs[f"{key}_{slot}"] = selected
                            if key == "s1" and selected:
                                orbit = selected.get("relativeOrbitNumber_start")
                    active = {k.rsplit("_", 1)[1] for k, v in inputs.items() if v}
                    latest = max((v["system:time_start"] for v in inputs.values() if v), default=0)
                    if len(active) < 2 or (dw["system:time_start"] - latest) / 86400000 > args.latest_days:
                        atomic_json(cache_file, None)
                        continue
                    sm = select_scene(ee, "sm", s, str(t - timedelta(days=args.sm_days)),
                                      str(t + timedelta(days=args.sm_days + 1)), str(t), 0.01)
                    if sm:
                        sm["offset_days"] = (datetime.fromtimestamp(sm["system:time_start"] / 1000, timezone.utc).date() - t).days
                    pml = select_scene(ee, "pml", s, str(t - timedelta(days=7)),
                                       str(t + timedelta(days=1)), str(t), 0.01)
                    if pml:
                        ps = datetime.fromtimestamp(pml["system:time_start"] / 1000, timezone.utc).date()
                        pe = min(ps + timedelta(days=8), date(ps.year + 1, 1, 1))
                        if not ps <= t < pe:
                            pml = None
                        else:
                            pml.update(interval_start=str(ps), interval_end=str(pe))
                    s.update(target_time=str(t), inputs=inputs, labels={"dw": dw, "sm": sm, "pml": pml})
                    s["sample_id"] = fingerprint(s)[:24]
                    atomic_json(cache_file, s)
                    rows.append(s)
                    location_counts[location] = location_counts.get(location, 0) + 1
                    occupied.setdefault(occupied_key, []).append((transform[2], transform[5]))
                    accepted += 1
                    LOG.info("Planned %s/%s", len(rows), args.samples)
                if accepted != required:
                    LOG.warning("Deficit month=%s stratum=%s requested=%s actual=%s", month, code, required, accepted)
    atomic_json(root / "sampling_report.json", {"requested": args.samples, "accepted": len(rows), "candidate_population": counts, "frequency_quotas": quota(args.samples, FREQUENCY_BINS), "sampling_strategy": SAMPLING_STRATEGY})
    if len(rows) < args.samples and not args.allow_shortfall:
        raise RuntimeError(f"Only {len(rows)}/{args.samples} samples; inspect sampling_report.json. "
                           "Use --allow-shortfall to explicitly accept the cached partial plan, "
                           "or use a new output directory with a larger candidate pool.")
    if not rows:
        raise RuntimeError("No eligible samples; inspect candidate coverage and thresholds")
    # Nested dictionaries serialized to JSON for a portable immutable Parquet manifest.
    rows = [dict(s, inputs=json.dumps(s["inputs"]), labels=json.dumps(s["labels"]),
                 transform=json.dumps(s["transform"])) for s in rows]
    tmp = Path(str(path) + ".part")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return rows


def export_image(ee, sample):
    chunks, names = [], []
    for slot in range(4):
        for key, bands in BANDS.items():
            scene = sample["inputs"][f"{key}_{slot}"]
            im = prepare(ee, key, ee.Image(scene["asset_id"])) if scene else ee.Image.constant([0] * len(bands)).rename(bands).updateMask(ee.Image(0))
            # Extend each band separately: the first scene's footprint must not
            # truncate other sensors or labels at a scene boundary.
            valid = im.mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False).gt(0)
            bn = [f"{key}_{slot}_{b}" for b in bands]
            chunks.extend([im.unmask(0, sameFootprint=False).rename(bn), valid.rename(f"{key}_{slot}_valid")])
            names.extend(bn + [f"{key}_{slot}_valid"])
    for key, name in [("dw", "water"), ("sm", "soil_moisture"), ("pml", "et")]:
        scene = sample["labels"][key]
        im = prepare(ee, key, ee.Image(scene["asset_id"])) if scene else ee.Image(0).rename(name).updateMask(ee.Image(0))
        chunks.extend([im.unmask(0, sameFootprint=False),
                       im.mask().unmask(0, sameFootprint=False).gt(0).rename(name + "_valid")])
        names.extend([name, name + "_valid"])
    return ee.Image.cat(chunks).toFloat().clip(region(ee, sample)), names


def drive_client(args):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    credentials = Credentials.from_authorized_user_file(args.drive_token)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Drive OAuth credentials are not valid")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_files(drive, query):
    found, token = [], None
    while True:
        result = drive.files().list(q=query, fields="nextPageToken,files(id,name,size,md5Checksum)",
                                    pageToken=token, pageSize=1000).execute(num_retries=5)
        found.extend(result.get("files", []))
        token = result.get("nextPageToken")
        if not token:
            return found


def download(drive, remote, path):
    from googleapiclient.http import MediaIoBaseDownload
    def valid(p):
        if not p.exists() or p.stat().st_size != int(remote["size"]):
            return False
        with p.open("rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest() == remote.get("md5Checksum")
    if valid(path):
        return
    temp = Path(str(path) + ".part")
    with temp.open("wb") as handle:
        dl = MediaIoBaseDownload(handle, drive.files().get_media(fileId=remote["id"]), chunksize=8 * 1024**2)
        done = False
        while not done:
            _, done = dl.next_chunk(num_retries=5)
        handle.flush()
        os.fsync(handle.fileno())
    if not valid(temp):
        raise RuntimeError(f"Download checksum/size mismatch: {path}")
    os.replace(temp, path)


def pml_native_targets(sample, et, valid):
    """Intersect complete native PML cells with the output grid in local equal-area CRS.

    Native boundaries are densified before reprojection. No nominal 500 m grid
    or rounded integer pooling factor is used. Returns one row per complete cell.
    """
    import numpy as np
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import Polygon, box
    from shapely.ops import transform
    from shapely.strtree import STRtree

    height, width = et.shape
    if height != width or height % 3:
        raise ValueError("Native PML supervision requires a square grid divisible by three")
    shape = (height // 3, width // 3)
    empty = dict(weights=np.zeros((1, *shape), np.float32),
                 value=np.zeros(1, np.float32), valid=np.zeros(1, np.uint8))
    scene = sample["labels"].get("pml")
    if not scene or not valid.any():
        return empty
    if "projection" not in scene:
        raise ValueError("PML native projection missing; refresh projection metadata before ingestion")
    native = Affine(*scene["projection"]["transform"])
    grid = Affine(*sample["transform"])
    area_crs = (f"+proj=laea +lat_0={sample['lat']} +lon_0={sample['lon']} "
                "+datum=WGS84 +units=m +type=crs")
    to_area = Transformer.from_crs(sample["crs"], area_crs, always_xy=True, force_over=True)
    native_area = Transformer.from_crs(scene["projection"]["crs"], area_crs, always_xy=True, force_over=True)
    to_native = Transformer.from_crs(sample["crs"], scene["projection"]["crs"], always_xy=True, force_over=True)
    to_grid = Transformer.from_crs(scene["projection"]["crs"], sample["crs"], always_xy=True, force_over=True)

    def polygon(affine, x, y, size, transformer):
        points = [affine * (x, y), affine * (x + size, y),
                  affine * (x + size, y + size), affine * (x, y + size)]
        poly = Polygon(points).segmentize(max(abs(affine.a), abs(affine.e)) * size / 16)
        return transform(transformer.transform, poly)

    footprint = polygon(grid, 0, 0, width, to_area)
    pixels = [polygon(grid, x, y, 3, to_area)
              for y in range(0, height, 3) for x in range(0, width, 3)]
    tree = STRtree(pixels)
    # Transform a densified window boundary to native pixel coordinates.
    boundary = box(0, 0, width, height).segmentize(1)
    positions = [~native * to_native.transform(*(grid * xy)) for xy in boundary.exterior.coords]
    low = np.floor(np.min(positions, axis=0)).astype(int)
    high = np.ceil(np.max(positions, axis=0)).astype(int)
    weights, values = [], []
    for y in range(low[1], high[1]):
        for x in range(low[0], high[0]):
            cell = polygon(native, x, y, 1, native_area)
            if cell.area <= 0 or cell.difference(footprint).area > cell.area * 1e-7:
                continue
            gx, gy = ~grid * to_grid.transform(*(native * (x + .5, y + .5)))
            col, row = int(np.floor(gx)), int(np.floor(gy))
            if not (0 <= row < height and 0 <= col < width and valid[row, col]):
                continue
            overlap = np.zeros(len(pixels), np.float64)
            for i in tree.query(cell, predicate="intersects"):
                overlap[i] = cell.intersection(pixels[i]).area
            if not np.isclose(overlap.sum(), cell.area, rtol=1e-6):
                raise ValueError("PML area weights do not cover the complete native cell")
            weights.append((overlap / overlap.sum()).reshape(shape))
            values.append(et[row, col])
    if not weights:
        return empty
    return dict(weights=np.asarray(weights, np.float32), value=np.asarray(values, np.float32),
                valid=np.ones(len(values), np.uint8))


def ingest(path, sample, names, root):
    import numpy as np
    import rasterio
    import zarr
    from importlib.metadata import version
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config
    from hydrostate.storage import SampleShardWriter, Schema
    schema = Schema()
    base = root / "zarr" / sample["sample_id"]
    inp, lab = base / "inputs.zarr", base / "labels.zarr"
    # Recover finalize() interrupted between its two atomic renames.
    for p in [inp, lab]:
        partial = Path(str(p) + ".partial")
        if not p.exists() and (partial / "_SUCCESS.json").exists():
            os.replace(partial, p)
    if all((p / "_SUCCESS.json").exists() for p in [inp, lab]):
        for p, specs in [(inp, schema.inputs), (lab, schema.targets)]:
            group = zarr.open_group(str(p), mode="r")
            if p == lab:
                if "et_native/0" not in group:
                    raise ValueError("Legacy labels lack PML weights; re-ingest into a new output directory")
                native = group["et_native/0"]
                weights, values, valid = (native[k][:] for k in ("weights", "value", "valid"))
                if (weights.shape != (len(values), SIZE // 3, SIZE // 3)
                        or valid.shape != values.shape or (weights < 0).any()
                        or not np.isfinite(weights).all() or not np.isfinite(values).all()
                        or not np.allclose(weights.sum(axis=(1, 2))[valid.astype(bool)], 1)):
                    raise ValueError("Invalid committed native PML supervision")
            for name, spec in specs.items():
                if group[name].shape != (1, *spec.tail_shape):
                    raise ValueError(f"Invalid committed array {p}/{name}")
                if not np.isfinite(group[name][:]).all():
                    raise ValueError(f"Nonfinite committed array {p}/{name}")
        return str(inp.relative_to(root)), str(lab.relative_to(root))
    # Incomplete, owned local stores can be recreated from the verified GeoTIFF.
    import shutil
    for p in [inp, lab]:
        partial = Path(str(p) + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
    with rasterio.open(path) as src:
        from rasterio.windows import Window
        from affine import Affine
        expected = Affine(*sample["transform"])
        col, row = (~src.transform) * (expected.c, expected.f)
        window = Window(round(col), round(row), SIZE, SIZE)
        # GEE can include a border pixel due to footprint reprojection rounding.
        # Read the exact manifest window; never resize or shift its pixel grid.
        if (src.count != len(names) or not SIZE <= src.width <= SIZE + 2
                or not SIZE <= src.height <= SIZE + 2 or min(window.col_off, window.row_off) < 0
                or window.col_off + SIZE > src.width or window.row_off + SIZE > src.height):
            raise ValueError(
                f"Unexpected raster shape: {path}; actual={src.height}x{src.width}, "
                f"bands={src.count}/{len(names)}, required={SIZE}x{SIZE}, "
                f"window_offset=({col}, {row}). Re-export if the window is truncated.")
        if src.crs.to_string() != sample["crs"] or not np.allclose(
                list(src.window_transform(window))[:6], sample["transform"], atol=1e-6, rtol=0):
            raise ValueError(f"Grid mismatch: {path}")
        if list(src.descriptions) != names:
            raise ValueError(f"Export band order mismatch: {path}")
        data = dict(zip(names, src.read(window=window)))
    inputs = {k: np.zeros(v.tail_shape, v.dtype) for k, v in schema.inputs.items()}
    targets = {k: np.zeros(v.tail_shape, v.dtype) for k, v in schema.targets.items()}
    norm = load_computed_config()
    for slot in range(4):
        for mi, (key, bands) in enumerate(BANDS.items()):
            valid = data[f"{key}_{slot}_valid"] > 0
            scene = sample["inputs"][f"{key}_{slot}"]
            inputs[f"{key}_valid"][slot, 0] = valid
            inputs["sensor_valid"][slot, mi] = bool(scene and valid.any())
            if scene:
                inputs["observation_time"][slot, mi] = scene["system:time_start"] // 1000
            for bi, band in enumerate(bands):
                canonical = band.lower() if key == "s1" else (
                    "B" + band[1:].zfill(2) if key == "s2" and band != "B8A" else band)
                stats = norm[MODALITIES[key]][canonical]
                value = (data[f"{key}_{slot}_{band}"] - stats["mean"]) / (4 * stats["std"]) + 0.5
                inputs[key][slot, bi] = np.where(valid, value, 0)
    for name in ["water", "soil_moisture", "et"]:
        targets[name][0] = data[name]
        targets[name + "_valid"][0] = data[name + "_valid"] > 0
    for values in [inputs, targets]:
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite data after conversion")
    if not np.isin(targets["water"], [0, 1]).all():
        raise ValueError("Water label must be binary")
    writer = SampleShardWriter(str(inp), str(lab), 1, attrs={
        "normalization": "OlmoEarth (x-mean)/(4*std)+0.5", "normalization_hash": fingerprint(norm),
        "olmoearth_pretrain_minimal": version("olmoearth-pretrain-minimal"),
        "sample": sample, "resolution": 10, "water_encoding": "dynamic_world_label_eq_0",
        "coarse_labels": "et_native_area_mean; smap_window_mean_weak",
        "et_temporal_support": "containing_8day_mean_daily_et"})
    writer.write(inputs, targets)
    native = pml_native_targets(sample, targets["et"][0], targets["et_valid"][0])
    group = writer.label_group.require_group("et_native/0")
    for key, value in native.items():
        group.create_dataset(key, data=value)
    writer.finalize()
    return ingest(path, sample, names, root)


def reingest_local(ee, sources, root):
    """Rebuild v3 shards from verified downloads; only fetch native grid metadata."""
    import pandas as pd
    entries, seen, projections = [], set(), {}
    for source in sources:
        if source.resolve() == root.resolve():
            raise ValueError("Re-ingestion requires a new output directory")
        for row in pd.read_parquet(source / "samples.parquet").to_dict("records"):
            sid = row["sample_id"]
            if sid in seen:
                raise ValueError(f"Duplicate sample_id in source manifests: {sid}")
            seen.add(sid)
            metadata = json.loads((source / "downloads" / f"{sid}.json").read_text())
            path = source / "downloads" / f"{sid}.tif"
            remote = metadata["remote"]
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "md5").hexdigest()
            if path.stat().st_size != int(remote["size"]) or checksum != remote["md5Checksum"]:
                raise ValueError(f"Local export checksum mismatch: {path}")
            sample = metadata["sample"]
            pml = sample["labels"].get("pml")
            if pml and "projection" not in pml:
                asset = pml["asset_id"]
                if asset not in projections:
                    projections[asset] = ee.Image(asset).select("ET").projection().getInfo()
                pml["projection"] = projections[asset]
            ip, lp = ingest(path, sample, metadata["bands"], root)
            entries.append(dict(row, input_shard=ip, input_row=0, label_shard=lp, label_row=0))
            LOG.info("Re-ingested %s", sid)
    destination = root / "samples.parquet"
    pd.DataFrame(entries).to_parquet(str(destination) + ".part", index=False)
    os.replace(str(destination) + ".part", destination)


def run(ee, args, root, rows):
    import pandas as pd
    drive = drive_client(args)
    folder_name = "HydroState_" + fingerprint(str(root))[:12]
    folders = list_files(drive, f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false")
    if len(folders) > 1:
        raise RuntimeError("Ambiguous Drive folder name")
    folder = folders[0]["id"] if folders else drive.files().create(
        body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()["id"]
    db = sqlite3.connect(root / "state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, task TEXT, status TEXT, remote TEXT, manifest TEXT, attempts INTEGER DEFAULT 0)")
    raw = root / "downloads"
    raw.mkdir(exist_ok=True)
    projections = {}
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        for row in batch:
            db.execute("INSERT OR IGNORE INTO tasks(id,status) VALUES (?, 'planned')", (row["sample_id"],))
        db.commit()
        while True:
            active = sum(bool(db.execute("SELECT task FROM tasks WHERE id=? AND task IS NOT NULL AND status NOT IN ('committed','cleaned')",
                             (r["sample_id"],)).fetchone()) for r in batch)
            all_done = True
            for row in batch:
                sid = row["sample_id"]
                task_id, status, remote_json, manifest = db.execute(
                    "SELECT task,status,remote,manifest FROM tasks WHERE id=?", (sid,)).fetchone()
                if status in ["committed", "cleaned"]:
                    continue
                all_done = False
                sample = dict(row)
                for key in ["inputs", "labels", "transform"]:
                    sample[key] = json.loads(sample[key])
                pml = sample["labels"].get("pml")
                if pml:
                    asset = pml["asset_id"]
                    if asset not in projections:
                        projections[asset] = ee.Image(asset).select("ET").projection().getInfo()
                    pml["projection"] = projections[asset]
                image, names = export_image(ee, sample)
                if not task_id:
                    if active >= args.concurrency:
                        continue
                    task = ee.batch.Export.image.toDrive(
                        image=image, description="hs_" + sid, folder=folder_name,
                        fileNamePrefix=sid, region=region(ee, sample), crs=sample["crs"],
                        crsTransform=sample["transform"], maxPixels=1000000, fileFormat="GeoTIFF")
                    task._request_id = ee.data.newTaskId()[0]
                    # Persist task ID before submission so restart cannot duplicate exports.
                    db.execute("UPDATE tasks SET task=?,status='submitted' WHERE id=?", (task._request_id, sid))
                    db.commit()
                    task.start()
                    task_id = task.id
                    db.execute("UPDATE tasks SET task=? WHERE id=?", (task_id, sid))
                    db.commit()
                    active += 1
                state = ee.data.getTaskStatus(task_id)[0]["state"]
                if state in ["READY", "RUNNING"]:
                    continue
                if state in ["UNKNOWN", "UNSUBMITTED"]:
                    matches = [t for t in ee.batch.Task.list()
                               if t.config.get("description") == "hs_" + sid
                               and t.state not in ["FAILED", "CANCELLED"]]
                    if matches:
                        db.execute("UPDATE tasks SET task=? WHERE id=?", (matches[0].id, sid))
                        db.commit()
                        continue
                    task = ee.batch.Export.image.toDrive(
                        image=image, description="hs_" + sid, folder=folder_name,
                        fileNamePrefix=sid, region=region(ee, sample), crs=sample["crs"],
                        crsTransform=sample["transform"], maxPixels=1000000, fileFormat="GeoTIFF")
                    task._request_id = task_id
                    task.start()
                    db.execute("UPDATE tasks SET task=? WHERE id=?", (task.id, sid))
                    db.commit()
                    continue
                if state in ["FAILED", "CANCELLED"]:
                    attempts = db.execute("SELECT attempts FROM tasks WHERE id=?", (sid,)).fetchone()[0]
                    error = ee.data.getTaskStatus(task_id)
                    if attempts >= 3:
                        raise RuntimeError(f"Export {sid} failed after 3 retries: {error}")
                    LOG.warning("Retry export %s: %s", sid, error)
                    db.execute("UPDATE tasks SET task=NULL,status='planned',attempts=attempts+1 WHERE id=?", (sid,))
                    db.commit()
                    active -= 1
                    continue
                if state != "COMPLETED":
                    raise RuntimeError(f"Unexpected task state: {state}")
                files = list_files(drive, f"'{folder}' in parents and name='{sid}.tif' and trashed=false")
                if not files:
                    continue
                if len(files) != 1:
                    raise RuntimeError(f"Duplicate export for {sid}")
                remote = files[0]
                db.execute("UPDATE tasks SET remote=?,status='exported' WHERE id=?", (json.dumps(remote), sid))
                db.commit()
                atomic_json(raw / f"{sid}.json", {"sample": sample, "bands": names, "remote": remote})
                local = raw / f"{sid}.tif"
                download(drive, remote, local)
                ip, lp = ingest(local, sample, names, root)
                entry = dict(row, input_shard=ip, input_row=0, label_shard=lp, label_row=0)
                db.execute("UPDATE tasks SET status='committed',manifest=? WHERE id=?", (json.dumps(entry), sid))
                db.commit()
                active -= 1
                LOG.info("Committed %s", sid)
            if all_done:
                break
            time.sleep(args.poll_seconds)
        # Publish manifest before remote deletion. Can be rebuilt from SQLite after a crash.
        records = [json.loads(r[0]) for r in db.execute("SELECT manifest FROM tasks WHERE manifest IS NOT NULL ORDER BY id")]
        destination = root / "samples.parquet"
        pd.DataFrame(records).to_parquet(str(destination) + ".part", index=False)
        os.replace(str(destination) + ".part", destination)
        for row in batch:
            sid = row["sample_id"]
            status, remote = db.execute("SELECT status,remote FROM tasks WHERE id=?", (sid,)).fetchone()
            if status == "cleaned":
                continue
            from googleapiclient.errors import HttpError
            try:
                drive.files().delete(fileId=json.loads(remote)["id"]).execute()
            except HttpError as exc:
                if exc.resp.status != 404:
                    raise
            db.execute("UPDATE tasks SET status='cleaned' WHERE id=?", (sid,))
            db.commit()
        LOG.info("Batch %s cleaned", offset // args.batch_size)
    db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--drive-token", help="Google authorized-user OAuth JSON with Drive scope")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--reingest-from", type=Path, action="append",
                   help="Existing run directory with verified GeoTIFFs; repeat to combine runs")
    p.add_argument("--allow-shortfall", action="store_true")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42, help="Sample selection/date seed")
    p.add_argument("--candidate-seed", type=int, default=42, help="Shared spatial pool seed")
    p.add_argument("--candidate-dir", type=Path, help="Shared spatial cache directory")
    p.add_argument("--climate-asset", default="users/xiaozhen6666666/cloud/Beck_KG_V1_present_0p0083")
    p.add_argument("--climate-band")
    p.add_argument("--candidates-per-class", type=int, default=16)
    p.add_argument("--max-dates-per-location", type=int, default=4)
    p.add_argument("--min-valid", type=float, default=0.5)
    p.add_argument("--latest-days", type=int, default=3)
    p.add_argument("--sm-days", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--poll-seconds", type=int, default=30)
    args = p.parse_args()
    if args.start_year > args.end_year or min(args.samples, args.batch_size, args.concurrency, args.candidates_per_class, args.max_dates_per_location, args.poll_seconds) < 1:
        p.error("Invalid years/counts")
    if not 0 < args.min_valid <= 1 or min(args.sm_days, args.latest_days) < 0:
        p.error("Invalid quality threshold or date tolerance")
    if not args.plan_only and not args.reingest_from and not args.drive_token:
        p.error("--drive-token is required for transfer")
    if args.plan_only and args.reingest_from:
        p.error("--plan-only and --reingest-from cannot be combined")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.candidate_dir = (args.candidate_dir or root.parent / "candidates").resolve()
    # Prevent concurrent writers and deletion races in this run directory.
    import fcntl
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ["reingest_from", "drive_token", "plan_only", "allow_shortfall", "poll_seconds", "concurrency", "batch_size"]}
        config["output"] = str(root)
        config["schema_version"] = "3.0"
        if args.reingest_from:
            config["reingest_from"] = [str(p.resolve()) for p in args.reingest_from]
        if not args.reingest_from:
            config["sampling_strategy"] = SAMPLING_STRATEGY
        cfg = root / "config.json"
        if cfg.exists():
            previous = json.loads(cfg.read_text())
            # Resume pre-sharing runs without changing their sampling sequence.
            previous.setdefault("candidate_seed", previous["seed"])
            previous.setdefault("candidate_dir", str((root / "candidates").resolve()))
            if previous != config:
                raise RuntimeError("Configuration changed: use a new output directory")
        atomic_json(cfg, config)
        if args.reingest_from:
            import ee
            ee.Initialize(project=args.project)
            reingest_local(ee, args.reingest_from, root)
            return
        if not args.plan_only:
            import rasterio  # noqa: F401
            import zarr  # noqa: F401
            from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config
            load_computed_config()
            drive_client(args)
        import ee
        ee.Initialize(project=args.project)
        LOG.info("Earth Engine project=%s; output=%s", args.project, root)
        rows = plan(ee, args, root)
        LOG.info("Loaded sampling plan: %s samples", len(rows))
        if not args.plan_only:
            run(ee, args, root, rows)


if __name__ == "__main__":
    main()
