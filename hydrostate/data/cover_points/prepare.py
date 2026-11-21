"""Parse published manual points and assign unique windows with a context halo."""

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.contract import PROFILE, product_end
from hydrostate.data.spatial import grid
from hydrostate.storage.packs import file_hash

from .contract import CLASS_MAP, CLASSES, VERSION, config_identity, producer_identity


def atomic_parquet(table, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".part")
    table.to_parquet(temp, index=False)
    temp.replace(path)


def parse_points(source):
    raw = pd.read_csv(source, dtype={"image_date": str})
    required = {"site_name", "image_date", "image_product", "class", "geometry"}
    if not required <= set(raw) or raw.empty:
        raise ValueError("Missing manual-point fields or empty source")
    if not raw.image_product.eq("Sentinel-2_TOA").all():
        raise ValueError("First release requires Sentinel-2_TOA manual training points")
    records = []
    for i, r in enumerate(raw.to_dict("records")):
        value = r["class"]
        if pd.isna(value) or value not in CLASS_MAP:
            raise ValueError(f"Unknown original class at source row {i}: {value}")
        geometry = wkt.loads(r["geometry"])
        if geometry.geom_type == "MultiPoint" and len(geometry.geoms) == 1:
            geometry = geometry.geoms[0]
        if geometry.geom_type != "Point" or geometry.is_empty:
            raise ValueError("Each annotation must contain exactly one point")
        lon, lat = geometry.x, geometry.y
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -60 <= lat < 84):
            raise ValueError("Point outside supported geographic input domain")
        day = pd.to_datetime(r["image_date"], format="%Y%m%d", errors="raise").date()
        if not 2017 <= day.year <= 2024:
            raise ValueError("Annotation outside current comparison years")
        records.append(
            dict(
                source_row=i,
                site_name=str(r["site_name"]),
                observation_date=day.isoformat(),
                lon=lon,
                lat=lat,
                original_class=int(value),
                cover_class=CLASS_MAP[value],
                source_role="training",
                evaluation_role="external_test",
            )
        )
    table = pd.DataFrame(records)
    key = ["site_name", "observation_date", "lon", "lat"]
    if table.groupby(key).original_class.nunique().gt(1).any():
        raise ValueError("Conflicting labels at identical point/date")
    duplicates = int(table.duplicated(key).sum())
    table = table.drop_duplicates(key).reset_index(drop=True)
    table["point_index"] = np.arange(len(table), dtype="int64")
    return table, duplicates


def plan_windows(points, identity):
    """900 m ownership cells within 1200 m windows; each point scored once."""
    rows, windows = [], {}
    for p in points.to_dict("records"):
        crs, _ = grid(p["lon"], p["lat"])
        forward = Transformer.from_crs(4326, crs, always_xy=True)
        x, y = forward.transform(p["lon"], p["lat"])
        left = float(np.floor(x / 900) * 900 - 150)
        top = float((np.floor(y / 900) + 1) * 900 + 150)
        transform = [10, 0, left, 0, -10, top]
        day = date.fromisoformat(p["observation_date"])
        start = date(day.year, 1, 1) + timedelta(days=((day.timetuple().tm_yday - 1) // 8) * 8)
        end = product_end(start.isoformat())
        sid = fingerprint(
            dict(
                crs=crs, transform=transform, target_time=start.isoformat(), config_sha256=identity
            )
        )[:24]
        row, col = int(np.floor((top - y) / 30)), int(np.floor((x - left) / 30))
        if not (5 <= row <= 35 and 5 <= col <= 35):
            raise ValueError("Point lost context halo")
        lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(left + 600, top - 600)
        windows[sid] = dict(
            sample_id=sid,
            version=VERSION,
            input_profile=PROFILE,
            split="test",
            crs=crs,
            transform=json.dumps(transform),
            lon=lon,
            lat=lat,
            target_time=start.isoformat(),
            target_end=end,
            block=json.dumps([int(np.floor(lon)), int(np.floor(lat)), 1]),
        )
        rows.append(
            dict(
                p,
                sample_id=sid,
                output_row=row,
                output_col=col,
                target_time=start.isoformat(),
                target_end=end,
                date_offset_days=(day - start).days,
            )
        )
    points = pd.DataFrame(rows)
    samples = pd.DataFrame(windows.values()).sort_values("sample_id").reset_index(drop=True)
    counts = points.groupby("sample_id").size()
    samples["observation_capacity"] = samples.sample_id.map(counts).astype(int)
    samples["site_name"] = samples.sample_id.map(
        points.groupby("sample_id").site_name.agg(lambda v: ";".join(sorted(set(v))))
    )
    return points, samples


def prepare(source, root, *, min_active_slots=2, min_input_fraction=0.5):
    root, source = Path(root), Path(source)
    root.mkdir(parents=True, exist_ok=True)
    points, duplicates = parse_points(source)
    config = dict(
        version=VERSION,
        input_profile=PROFILE,
        source_sha256=file_hash(source),
        producer_identity=producer_identity(),
        classes=list(CLASSES),
        class_map={str(k): v for k, v in CLASS_MAP.items()},
        calendar="Jan01_8day_target_interval",
        point_time_precision="date",
        ownership_metres=900,
        halo_metres=150,
        min_active_slots=min_active_slots,
        min_input_fraction=min_input_fraction,
    )
    if not (1 <= min_active_slots <= 5 and 0 < min_input_fraction <= 1):
        raise ValueError("Invalid input eligibility thresholds")
    identity = config_identity(config)
    points, samples = plan_windows(points, identity)
    with locked(root / ".collection.lock", blocking=False):
        path = root / "collection.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError("Source or collection semantics changed; use a new root")
            for name, table in [("points.parquet", points), ("plan.parquet", samples)]:
                if not pd.read_parquet(root / name).equals(table):
                    raise ValueError("Existing plan/points differ; do not overwrite")
            return json.loads((root / "audit" / "preparation.json").read_text())
        (root / "sources").mkdir(exist_ok=True)
        (root / "audit").mkdir(exist_ok=True)
        shutil.copyfile(source, root / "sources" / source.name)
        atomic_parquet(points, root / "points.parquet")
        atomic_parquet(samples, root / "plan.parquet")
        atomic_json(path, config)
        report = dict(
            points=len(points),
            windows=len(samples),
            duplicates_removed=duplicates,
            sites=sorted(points.site_name.unique()),
            site_dates=len(points[["site_name", "observation_date"]].drop_duplicates()),
            class_counts={
                CLASSES[int(k)]: int(v) for k, v in points.cover_class.value_counts().items()
            },
            independent_evaluation="requires_checkpoint_training_overlap_audit",
        )
        atomic_json(root / "audit" / "preparation.json", report)
        return report
