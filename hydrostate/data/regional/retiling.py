"""Retile prepared inputs offline, referencing immutable native arrays without copying."""

import copy
import json
from functools import cache
from pathlib import Path

import pandas as pd
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely import STRtree
from shapely.geometry import shape
from shapely.ops import transform

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.regional.exports import catalog
from hydrostate.data.regional.plan import (
    blocks_for_window,
    load_plan,
    product,
    window_box,
)
from hydrostate.data.retrieval.composites import radar_orbit
from hydrostate.data.retrieval.contract import INPUT_GROUPS


def retile(root, source_root, dates, *, core=36, stride=34):
    """Keep the source AOI/grid and recompute each new window's S1 orbit from metadata.

    The last row/column is anchored to the source boundary when necessary; only
    that outermost overlap can exceed the requested overlap. Where a new context
    lacks cached inputs, existing source windows fill coverage gaps. No network calls.
    """
    root, source_root = Path(root), Path(source_root).resolve()
    if root.resolve() == source_root or not dates:
        raise ValueError("Retiling requires a separate root and explicit ready dates")
    if not 2 <= core <= 40 or core % 2 or not 0 < stride <= core or stride % 2:
        raise ValueError("Retained core and stride must be even and fit the 40-pixel output")
    source = load_plan(source_root)
    if source.get("input_source"):
        raise ValueError("Retile from the original prepared input root, not another retiled view")
    if not set(dates) <= {p["start"] for p in source["periods"]}:
        raise ValueError("Dates must belong to the source plan")
    plan = copy.deepcopy(source)
    plan.pop("identity")
    plan["config"].update(core_size=core, stride=stride)
    plan["input_source"] = dict(
        root=str(source_root), identity=source["identity"], mode="direct_arrays",
        window_policy="regular_then_supported_source_windows_v1",
    )
    plan["identity"] = fingerprint(plan)
    geometry = shape(plan["geometry"])
    step = stride * 3

    def positions(size):
        last = size - 120
        return sorted(set([*range(0, last + 1, step), last]))

    windows = [dict(row=y, col=x)
               for y in positions(plan["grid"]["height"])
               for x in positions(plan["grid"]["width"])
               if geometry.intersects(window_box(plan, y, x).buffer(-(40 - core) * 15))]
    project = Transformer.from_crs(plan["grid"]["crs"], 4326, always_xy=True).transform
    footprints = [transform(project, window_box(plan, r["row"], r["col"])) for r in windows]
    grid = plan["grid"]
    aoi = geometry_mask(
        [plan["geometry"]], (grid["height"] // 3, grid["width"] // 3),
        Affine(30, 0, grid["left"], 0, -30, grid["top"]), invert=True,
    )
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "prepare.lock", blocking=False):
        if (root / "plan.json").exists() and load_plan(root) != plan:
            raise ValueError("Existing retiled plan differs")
        # Inference checks output disk headroom using these runtime settings too.
        for name in ("drive.json", "pixels.json"):
            origin = source_root / name
            if origin.exists() and not (root / name).exists():
                atomic_json(root / name, json.loads(origin.read_text()))
        if not (root / "drive.json").exists():
            raise FileNotFoundError("Source lacks the inference disk-reserve settings: drive.json")
        atomic_json(root / "plan.json", plan)
        pd.DataFrame(windows).to_parquet(root / "windows.parquet", index=False)
        for name in ("targets", "manifests"):
            (root / name).mkdir(exist_ok=True)
        for period in plan["periods"]:
            day = period["start"]
            if day not in dates or (root / "manifests" / f"{day}.json").exists():
                continue
            source_manifest = json.loads((source_root / "manifests" / f"{day}.json").read_text())
            if source_manifest["plan"] != source["identity"]:
                raise ValueError("Source manifest identity mismatch")
            available = {p["id"] for p in source_manifest["products"]}
            scenes = catalog(None, source_root, source, "s1", period["slots"][0][0],
                             period["slots"][-1][1])
            tree = STRtree([shape(s["geometry"]) for s in scenes])
            rows, demands = [], {}
            remaining = aoi.copy()

            @cache
            def block_products(block, orbit):
                return [product(source, name, slot, block,
                                orbit if spec.sensor == "s1" else None)
                        for name, spec in INPUT_GROUPS.items() for slot in period["slots"]]

            def region(window):
                y = window["row"] // 3 + (40 - core) // 2
                x = window["col"] // 3 + (40 - core) // 2
                return slice(y, y + core), slice(x, x + core)

            def add(window, orbit):
                items = [p for block in blocks_for_window(plan, window["row"], window["col"])
                         for p in block_products(block, orbit)]
                if any(p["id"] not in available for p in items):
                    return False
                rows.append(dict(row=window["row"], col=window["col"], orbit=json.dumps(orbit)))
                demands.update((p["id"], p) for p in items)
                remaining[region(window)] = False
                return True

            for window, footprint in zip(windows, footprints):
                indices = tree.query(footprint, predicate="intersects")
                orbit = radar_orbit([scenes[int(i)]["properties"] for i in indices])
                add(window, orbit)
            selected_regular = len(rows)
            if remaining.any():
                originals = pd.read_parquet(source_root / "targets" / f"{day}.parquet")
                for window in originals.to_dict("records"):
                    if remaining[region(window)].any():
                        orbit = json.loads(window["orbit"])
                        add(window, tuple(orbit) if orbit is not None else None)
            if remaining.any():
                raise ValueError(f"Prepared source cannot cover {int(remaining.sum())} AOI pixels")
            record = dict(
                plan=plan["identity"], period=period, products=list(demands.values()),
                retile=dict(regular_candidates=len(windows), selected_regular=selected_regular,
                            source_windows_added=len(rows) - selected_regular, uncovered_pixels=0),
            )
            pd.DataFrame(rows).to_parquet(root / "targets" / f"{day}.parquet", index=False)
            atomic_json(root / "targets" / f"{day}.json", record)
            atomic_json(root / "manifests" / f"{day}.json", record)
    return dict(root=str(root), plan=plan["identity"], windows=len(windows), dates=dates,
                window_counts={day: len(pd.read_parquet(root / "targets" / f"{day}.parquet"))
                               for day in dates})
