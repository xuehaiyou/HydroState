"""Freeze manual sources, native footprints, windows and region-level splits."""

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds

from hydrostate.data.cover_points.prepare import atomic_parquet
from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.retrieval.contract import PROFILE, product_end
from hydrostate.data.spatial import grid
from hydrostate.storage.packs import file_hash

from .contract import VERSION, producer_identity
from .splits import assign


def flood_catalog(source, root):
    source = Path(source)
    metadata = json.loads((source / "Sen1Floods11_Metadata.geojson").read_text())
    events = {r["properties"]["location"]: r["properties"] for r in metadata["features"]}
    folder = root / "sources" / "LabelHand"
    folder.mkdir(parents=True)
    shutil.copyfile(
        source / "Sen1Floods11_Metadata.geojson", root / "sources" / "Sen1Floods11_Metadata.geojson"
    )
    rows = []
    for p in sorted((source / "data/flood_events/HandLabeled/LabelHand").glob("*_LabelHand.tif")):
        event = p.name.split("_")[0]
        # The released chip prefix is Mekong; event metadata names it Cambodia.
        e = events[{"Mekong": "Cambodia"}.get(event, event)]
        day = e["s2_date"].replace("/", "-")
        with rasterio.open(p) as src:
            values = src.read(1)
            if src.crs is None or not np.isin(values, [-1, 0, 1]).all():
                raise ValueError(f"Invalid manual label {p}")
            bounds = list(src.bounds)
            crs, affine = src.crs.to_string(), list(src.transform)[:6]
            counts = {str(k): int((values == k).sum()) for k in [-1, 0, 1]}
        shutil.copyfile(p, folder / p.name)
        rows.append(
            dict(
                event=event,
                chip_id=p.stem.removesuffix("_LabelHand"),
                path=f"sources/LabelHand/{p.name}",
                size_bytes=p.stat().st_size,
                observation_date=day,
                s1_date=e["s1_date"].replace("/", "-"),
                s2_date=day,
                included=2017 <= date.fromisoformat(day).year <= 2024,
                exclusion_reason=""
                if 2017 <= date.fromisoformat(day).year <= 2024
                else "outside_2017_2024_input_contract",
                crs=crs,
                transform=affine,
                bounds=bounds,
                counts=counts,
            )
        )
    if not rows:
        raise ValueError("No LabelHand sources")
    atomic_json(root / "sources/flood_catalog.json", rows)
    return rows


def flood_plan(catalog):
    rows = []
    table = pd.DataFrame([r for r in catalog if r["included"]])
    for event, chips in table.groupby("event"):
        west = min(r[0] for r in chips.bounds)
        east = max(r[2] for r in chips.bounds)
        south = min(r[1] for r in chips.bounds)
        north = max(r[3] for r in chips.bounds)
        crs, _ = grid((west + east) / 2, (south + north) / 2)
        centres = Transformer.from_crs(crs, 4326, always_xy=True)
        cells = set()
        for chip in chips.to_dict("records"):
            left, bottom, right, top = transform_bounds(
                chip["crs"], crs, *chip["bounds"], densify_pts=21
            )
            cells.update(
                (x, y)
                for x in range(int(np.floor(left / 1200)), int(np.ceil(right / 1200)))
                for y in range(int(np.floor(bottom / 1200)), int(np.ceil(top / 1200)))
            )
        day = date.fromisoformat(chips.iloc[0].observation_date)
        start = date(day.year, 1, 1) + timedelta(days=((day.timetuple().tm_yday - 1) // 8) * 8)
        for x, y in sorted(cells):
            tr = [10, 0, float(x * 1200), 0, -10, float((y + 1) * 1200)]
            lon, lat = centres.transform(tr[2] + 600, tr[5] - 600)
            sid = fingerprint(
                dict(
                    version=VERSION,
                    source="sen1floods11",
                    event=event,
                    crs=crs,
                    transform=tr,
                    target_time=start.isoformat(),
                )
            )[:24]
            rows.append(
                dict(
                    sample_id=sid,
                    source="sen1floods11",
                    region="flood:" + event,
                    event=event,
                    crs=crs,
                    transform=json.dumps(tr),
                    lon=lon,
                    lat=lat,
                    target_time=start.isoformat(),
                    target_end=product_end(start.isoformat()),
                    observation_date=day.isoformat(),
                    s1_date=chips.iloc[0].s1_date,
                    s2_date=day.isoformat(),
                    date_offset_days=(day - start).days,
                    block=json.dumps([int(np.floor(lon)), int(np.floor(lat)), 1]),
                    input_profile=PROFILE,
                    version=VERSION,
                )
            )
    return pd.DataFrame(rows)


def prepare(root, glacier_root, flood_root):
    root, glacier_root = Path(root), Path(glacier_root).resolve()
    if root.exists():
        raise FileExistsError("Preparation is immutable; use a new root or resume later stages")
    publication = json.loads((glacier_root / "_PUBLISHED.json").read_text())
    for name, key in [("samples.parquet", "samples_sha256"), ("points.parquet", "points_sha256")]:
        if file_hash(glacier_root / name) != publication[key]:
            raise ValueError("Published glacier source changed")
    root.mkdir(parents=True)
    (root / "audit").mkdir()
    catalog = flood_catalog(flood_root, root)
    points = pd.read_parquet(glacier_root / "points.parquet")
    glacier = pd.read_parquet(glacier_root / "samples.parquet").copy()
    glacier["source_sample_id"] = glacier.sample_id
    glacier["source"] = "glacier"
    glacier["region"] = "glacier:" + glacier.site_name
    glacier["version"] = VERSION
    glacier["event"] = glacier.site_name
    glacier["sample_id"] = glacier.source_sample_id.map(lambda s: fingerprint([VERSION, s])[:24])
    mapping = dict(zip(glacier.source_sample_id, glacier.sample_id))
    points["source_sample_id"] = points.sample_id
    points["sample_id"] = points.sample_id.map(mapping)
    if points.sample_id.isna().any():
        raise ValueError("Glacier input missing; resolve source rejections before reuse")
    # Different source dates in the same output cell remain conflict candidates.
    plan = pd.concat([glacier, flood_plan(catalog)], ignore_index=True)
    plan = plan.drop(columns=["input_path", "label_path", "shard_index", "split"], errors="ignore")
    plan, distances = assign(plan)
    config = dict(
        version=VERSION,
        input_profile=PROFILE,
        producer_identity=producer_identity(),
        glacier_root=str(glacier_root),
        glacier_publication=publication,
        min_active_slots=2,
        min_input_fraction=0.5,
        concurrency=4,
        point_assumption="annotated_30m_pixel_is_pure_user_decision",
        flood_time_anchor="s2_date_with_s1_date_preserved_manual_multisensor_labels",
        flood_geometry="union_original_pixel_polygons_then_utm_area_intersection",
        flood_full_valid_tolerance=1e-5,
        split_margin_m=35000,
        flood_class_threshold=0.5,
        fractions_units="1",
        calendar="Jan01_8day_target_interval",
    )
    atomic_parquet(points, root / "points.parquet")
    atomic_parquet(plan, root / "plan.parquet")
    atomic_parquet(distances, root / "audit/region_distances.parquet")
    atomic_parquet(
        plan[["sample_id", "source", "region", "spatial_group", "split"]], root / "splits.parquet"
    )
    atomic_json(root / "collection.json", config)
    report = dict(
        windows=len(plan),
        glacier_points=len(points),
        flood_chips=len(catalog),
        included_flood_chips=sum(r["included"] for r in catalog),
        region_splits=plan.groupby(["source", "region", "split"])
        .size()
        .reset_index(name="planned_windows")
        .to_dict("records"),
    )
    atomic_json(root / "audit/preparation.json", report)
    return report
