"""Retile prepared inputs offline, linking immutable native arrays without copying."""

import copy
import json
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import shape
from shapely.ops import transform

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.regional.exports import catalog
from hydrostate.data.regional.plan import (
    block_grid,
    blocks_for_window,
    load_plan,
    product,
    window_box,
)
from hydrostate.data.retrieval.composites import radar_orbit
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.storage.regional import product_path, ready


def retile(root, source_root, dates, *, core=36, stride=34):
    """Keep the source AOI/grid and recompute each new window's S1 orbit from metadata.

    The last row/column is anchored to the source boundary when necessary; only
    that outermost overlap can exceed the requested overlap. No network calls.
    """
    root, source_root = Path(root), Path(source_root).resolve()
    if root.resolve() == source_root or not dates:
        raise ValueError("Retiling requires a separate root and explicit ready dates")
    if not 2 <= core <= 40 or core % 2 or not 0 < stride <= core or stride % 2:
        raise ValueError("Retained core and stride must be even and fit the 40-pixel output")
    source = load_plan(source_root)
    if not set(dates) <= {p["start"] for p in source["periods"]}:
        raise ValueError("Dates must belong to the source plan")
    plan = copy.deepcopy(source)
    plan.pop("identity")
    plan["config"].update(core_size=core, stride=stride)
    plan["input_source"] = dict(
        root=str(source_root), identity=source["identity"], mode="linked_arrays",
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
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "prepare.lock", blocking=False):
        if (root / "plan.json").exists() and load_plan(root) != plan:
            raise ValueError("Existing retiled plan differs")
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
            scenes = catalog(None, source_root, source, "s1", period["slots"][0][0],
                             period["slots"][-1][1])
            tree = STRtree([shape(s["geometry"]) for s in scenes])
            rows, demands = [], {}
            for window, footprint in zip(windows, footprints):
                indices = tree.query(footprint, predicate="intersects")
                orbit = radar_orbit([scenes[int(i)]["properties"] for i in indices])
                rows.append(dict(window, orbit=json.dumps(orbit)))
                for block in blocks_for_window(plan, window["row"], window["col"]):
                    for name, spec in INPUT_GROUPS.items():
                        chosen = orbit if spec.sensor == "s1" else None
                        for slot in period["slots"]:
                            item = product(plan, name, slot, block, chosen)
                            demands[item["id"]] = item
            for item in demands.values():
                if ready(root, item):
                    continue
                old = product(source, item["group"], item["slot"], item["block"], item["orbit"])
                if not ready(source_root, old):
                    raise FileNotFoundError(f"New window needs an uncached product: {old['id']}")
                origin, destination = product_path(source_root, old), product_path(root, item)
                destination.mkdir(parents=True, exist_ok=True)
                atomic_json(destination / ".zgroup", dict(zarr_format=2))
                atomic_json(destination / ".zattrs", dict(
                    identity=item, input_contract=plan["input_contract"],
                    grid=block_grid(plan, item["block"], INPUT_GROUPS[item["group"]].resolution),
                ))
                for field in ("values", "valid", "quality"):
                    link = destination / field
                    if not link.is_symlink():
                        link.symlink_to(origin / field, target_is_directory=True)
                    if link.resolve() != (origin / field).resolve() or not link.exists():
                        raise ValueError("Invalid native array link")
                atomic_json(destination / "_REUSE.json", dict(source_root=str(source_root),
                            product=old, mode="linked_arrays"))
                atomic_json(destination / "_SUCCESS.json", item)
            record = dict(plan=plan["identity"], period=period, products=list(demands.values()))
            pd.DataFrame(rows).to_parquet(root / "targets" / f"{day}.parquet", index=False)
            atomic_json(root / "targets" / f"{day}.json", record)
            atomic_json(root / "manifests" / f"{day}.json", record)
    return dict(root=str(root), plan=plan["identity"], windows=len(windows), dates=dates)
