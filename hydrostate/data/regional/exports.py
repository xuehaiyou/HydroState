"""Metadata-only planning and shared GEE composite expressions for Drive exports."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import box, shape
from shapely.ops import transform

from hydrostate.data.gee.backends.drive import DriveTransport
from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.regional.plan import block_grid, blocks_for_window, epoch, product, window_box
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.composites import (
    QUALITY,
    radar_orbit,
    scene_catalog,
    sensor_composite,
)
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.storage.regional import ready


def catalog(ee, root, plan, sensor, start, end):
    """Cache monthly metadata, then use spatial indices locally (no per-window QA RPC)."""
    grid = plan["grid"]
    bounds = [
        grid["left"],
        grid["top"] - grid["height"] * 10,
        grid["left"] + grid["width"] * 10,
        grid["top"],
    ]
    area = ee.Geometry.Rectangle(bounds, grid["crs"], False)
    month = date.fromisoformat(start).replace(day=1)
    rows = []
    while month < date.fromisoformat(end):
        following = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        key = fingerprint([plan["identity"], sensor, str(month)])
        rows.extend(
            cached_json(
                Path(root) / "catalog" / f"{key}.json",
                lambda: scene_catalog(ee, sensor, area, str(month), str(following)),
            )
        )
        month = following
    return [
        r
        for r in rows
        if epoch(start) * 1000 <= r["properties"]["system:time_start"] < epoch(end) * 1000
    ]


def jobs(ee, root, plan, settings, reserved=()):
    """Yield bounded jobs lazily; each reusable product is scheduled only once."""
    root = Path(root)
    folder = root / "targets"
    folder.mkdir(exist_ok=True)
    windows = pd.read_parquet(root / "windows.parquet").to_dict("records")
    to_geo = Transformer.from_crs(plan["grid"]["crs"], 4326, always_xy=True).transform
    footprints = [transform(to_geo, window_box(plan, r["row"], r["col"])) for r in windows]
    seen = set(reserved)
    for period in plan["periods"]:
        if (root / "manifests" / (period["start"] + ".json")).exists():
            continue
        scenes = {
            s: catalog(ee, root, plan, s, period["slots"][0][0], period["slots"][-1][1])
            for s in ("s1", "s2", "landsat")
        }
        trees = {s: STRtree([shape(r["geometry"]) for r in rows]) for s, rows in scenes.items()}
        demands, rows = {}, []
        for window, footprint in zip(windows, footprints):
            indices = trees["s1"].query(footprint, predicate="intersects")
            orbit = radar_orbit([scenes["s1"][int(i)]["properties"] for i in indices])
            rows.append(dict(window, orbit=json.dumps(orbit)))
            for block in blocks_for_window(plan, window["row"], window["col"]):
                for name, group in INPUT_GROUPS.items():
                    selected_orbit = orbit if group.sensor == "s1" else None
                    key = (name, block, selected_orbit)
                    if key not in demands:
                        demands[key] = [
                            product(plan, name, slot, block, selected_orbit)
                            for slot in period["slots"]
                        ]
        all_products = {p["id"]: p for products in demands.values() for p in products}
        target = dict(period=period, products=list(all_products.values()), plan=plan["identity"])
        path = folder / (period["start"] + ".json")
        if path.exists() and json.loads(path.read_text()) != target:
            raise ValueError("Frozen regional source selection changed")
        table_path = path.with_suffix(".parquet")
        if not table_path.exists():
            temporary = table_path.with_suffix(".parquet.part")
            pd.DataFrame(rows).to_parquet(temporary, index=False)
            temporary.replace(table_path)
        elif pd.read_parquet(table_path).to_dict("records") != rows:
            raise ValueError("Frozen per-window radar selection changed")
        atomic_json(path, target)
        for (name, block, orbit), products in demands.items():
            spec = INPUT_GROUPS[name]
            grid = block_grid(plan, block, spec.resolution)
            names = [*spec.bands, *QUALITY]
            bytes_per_slot = grid["height"] * grid["width"] * len(names) * 4
            capacity = min(5, int(settings["max_export_gib"] * 1024**3 // bytes_per_slot))
            if capacity < 1:
                raise ValueError(
                    "Block exceeds export budget; initialize with a smaller block_size"
                )
            missing = [p for p in products if p["id"] not in seen and not ready(root, p)]
            tr = grid["transform"]
            area = transform(
                to_geo,
                box(
                    tr[2],
                    tr[5] - grid["height"] * spec.resolution,
                    tr[2] + grid["width"] * spec.resolution,
                    tr[5],
                ),
            )
            indices = trees[spec.sensor].query(area, predicate="intersects")
            available = [scenes[spec.sensor][int(i)]["properties"] for i in indices]
            for offset in range(0, len(missing), capacity):
                selected = missing[offset : offset + capacity]
                ids = []
                for p in selected:
                    ids.append(
                        sorted(
                            s["asset_id"]
                            for s in available
                            if epoch(p["slot"][0]) * 1000
                            <= s["system:time_start"]
                            < epoch(p["slot"][1]) * 1000
                            and (
                                spec.sensor != "s1"
                                or orbit is not None
                                and (
                                    s.get("relativeOrbitNumber_start"),
                                    s.get("orbitProperties_pass"),
                                )
                                == orbit
                            )
                        )
                    )
                job = dict(
                    products=selected,
                    group=name,
                    grid=grid,
                    scenes=ids,
                    bands=[f"t{i}_{band}" for i in range(len(selected)) for band in names],
                    raw_bytes=bytes_per_slot * len(selected),
                )
                job["id"] = fingerprint(job)[:24]
                seen.update(p["id"] for p in selected)
                yield job


class CompositeDriveTransport(DriveTransport):
    """Reuse verified downloads/cleanup; only the exported expression is different."""

    def submit(self, job, task_id):
        ee, spec, grid = self.ee, INPUT_GROUPS[job["group"]], job["grid"]
        images = []
        for index, (item, ids) in enumerate(zip(job["products"], job["scenes"])):
            images.append(
                sensor_composite(ee, spec.sensor, ids, item["slot"][0], spec.bands).rename(
                    [f"t{index}_{b}" for b in (*spec.bands, *QUALITY)]
                )
            )
        tr = grid["transform"]
        region = ee.Geometry.Rectangle(
            [
                tr[2],
                tr[5] - grid["height"] * spec.resolution,
                tr[2] + grid["width"] * spec.resolution,
                tr[5],
            ],
            grid["crs"],
            False,
        )
        task = ee.batch.Export.image.toDrive(
            image=ee.Image.cat(images),
            description=self.prefix(job["id"]),
            folder=self.folder_name,
            fileNamePrefix=self.prefix(job["id"]),
            region=region,
            crs=grid["crs"],
            crsTransform=tr,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            fileDimensions=self.settings["file_dimensions"],
            shardSize=256,
            skipEmptyTiles=False,
        )
        task._request_id = task_id
        task.start()
        return task.id
