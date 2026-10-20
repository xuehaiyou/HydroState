"""Pilot teacher-product consistency at the original observational supports.

This is not independent accuracy validation. References never select model inputs.
Only interior windows are used so the complete reference footprint is in the mosaic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from pyproj import Transformer

from hydrostate.data import gee_sampling as gs
from hydrostate.data.gee.client import retry
from hydrostate.inference.planning import input_transform


def get_reference(root, plan, period, row, ee):
    path = root / "references" / period["time"] / (row["window_id"] + ".zarr")
    group = zarr.open_group(str(path), mode="a")
    if group.attrs.get("complete"):
        return group
    grid = plan["grid"]
    tr = input_transform(grid)
    left = tr.c + row["col"] * 30
    top = tr.f - row["row"] * 30
    lon, lat = Transformer.from_crs(grid["crs"], 4326, always_xy=True).transform(
        left + 600, top - 600
    )
    sample = dict(crs=grid["crs"], transform=[10, 0, left, 0, -10, top], lon=lon, lat=lat)
    day = date.fromisoformat(period["time"])
    scenes = {}
    for key, minimum, before, after in (("dw", 0.5, 3, 4), ("sm", 0.01, 3, 4), ("pml", 0.01, 7, 1)):
        scenes[key] = retry(
            lambda key=key, minimum=minimum, before=before, after=after: gs.select_scene(
                ee,
                key,
                sample,
                str(day - timedelta(days=before)),
                str(day + timedelta(days=after)),
                str(day),
                minimum,
            )
        )
    if scenes["pml"]:
        scenes["pml"]["projection"] = (
            ee.Image(scenes["pml"]["asset_id"]).select("ET").projection().getInfo()
        )
    images, names = [], []
    for key, name in (("dw", "water"), ("sm", "soil_moisture"), ("pml", "et")):
        scene = scenes[key]
        image = (
            gs.prepare(ee, key, ee.Image(scene["asset_id"]))
            if scene
            else ee.Image(0).rename(name).updateMask(ee.Image(0))
        )
        images.extend(
            [
                image.unmask(0, sameFootprint=False).rename(name),
                image.mask().unmask(0, sameFootprint=False).rename(name + "_valid"),
            ]
        )
        names.extend([name, name + "_valid"])
    result = retry(
        lambda: ee.data.computePixels(
            dict(
                expression=ee.Image.cat(images).toFloat(),
                fileFormat="NUMPY_NDARRAY",
                bandIds=names,
                grid=dict(
                    dimensions=dict(width=120, height=120),
                    crsCode=grid["crs"],
                    affineTransform=dict(
                        scaleX=10, shearX=0, translateX=left, shearY=0, scaleY=-10, translateY=top
                    ),
                ),
            )
        )
    )
    if result.shape != (120, 120) or list(result.dtype.names or ()) != names:
        raise ValueError("Unexpected reference grid/bands")
    sample["labels"] = scenes
    native = gs.pml_native_targets(sample, result["et"], result["et_valid"] > 0)
    compressor = Blosc(cname="zstd", clevel=3)
    for name in names:
        group.array(name, result[name], chunks=(120, 120), compressor=compressor, overwrite=True)
    for name, values in native.items():
        group.array("et_native_" + name, values, compressor=compressor, overwrite=True)
    group.attrs.update(complete=True, scenes=scenes, sample=sample)
    return group


def metrics(pairs):
    if not pairs:
        return dict(count=0)
    values = np.asarray(pairs, "f8")
    pred, target = values[:, 0], values[:, 1]
    error = pred - target
    variance = np.square(target - target.mean()).sum()
    return dict(
        count=len(error),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        bias=float(error.mean()),
        r2=float(1 - np.square(error).sum() / variance) if variance > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--drop-s2", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--paired", action="store_true", help="Use valid support common to baseline and S2 ablation"
    )
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    index, period = next((i, p) for i, p in enumerate(plan["periods"]) if p["time"] == args.date)
    suffix = "_without_s2" if args.drop_s2 else ""
    product = zarr.open_group(str(root / "products" / f"poyang_3vars{suffix}.zarr"), mode="r")
    if not product["date_complete"][index]:
        raise ValueError("Prediction date is incomplete")
    other = None
    if args.paired:
        other_suffix = "" if args.drop_s2 else "_without_s2"
        other = zarr.open_group(
            str(root / "products" / f"poyang_3vars{other_suffix}.zarr"), mode="r"
        )
        if not other["date_complete"][index]:
            raise ValueError("Paired prediction date is incomplete")
    table = pd.read_parquet(root / "manifests" / f"{args.date}.parquet")
    grid, halo = plan["grid"], plan["grid"]["halo"]
    table = table.loc[
        table.accepted
        & (table.row >= halo)
        & (table.col >= halo)
        & (table.row + 40 - halo <= grid["height"])
        & (table.col + 40 - halo <= grid["width"])
    ]
    selected = table.iloc[
        np.unique(np.linspace(0, len(table) - 1, min(args.windows, len(table))).astype(int))
    ]
    ee = None
    if not args.offline:
        import ee

        ee.Initialize(project=plan["config"]["gee_project"])
        ee.data.setDeadline(180000)
    pairs = {k: [] for k in ("water", "soil_moisture", "et")}
    details = []
    for row in selected.to_dict("records"):
        if args.offline:
            reference = zarr.open_group(
                str(root / "references" / args.date / (row["window_id"] + ".zarr")), mode="r"
            )
            if not reference.attrs.get("complete"):
                raise ValueError("Reference incomplete")
        else:
            reference = get_reference(root, plan, period, row, ee)
        y, x = row["row"] - halo, row["col"] - halo
        predictions = {k: product[k][index, y : y + 40, x : x + 40] for k in pairs}
        if other is not None:
            for key in predictions:
                predictions[key][~np.isfinite(other[key][index, y : y + 40, x : x + 40])] = np.nan
        water = reference["water"][:].reshape(40, 3, 40, 3).mean((1, 3))
        valid = reference["water_valid"][:].reshape(40, 3, 40, 3).min((1, 3)) > 0
        selected_pixels = valid & np.isfinite(predictions["water"])
        pairs["water"].extend(zip(predictions["water"][selected_pixels], water[selected_pixels]))
        # Window land-weighted SM consistency, explicitly not full SMAP support.
        land = np.where(valid, 1 - water, 0)
        land[land <= 0.01] = 0
        if (
            valid.all()
            and reference["soil_moisture_valid"][60, 60] > 0
            and land.sum() > 0
            and np.isfinite(predictions["soil_moisture"][land > 0]).all()
        ):
            predicted = float(
                (np.nan_to_num(predictions["soil_moisture"]) * land).sum() / land.sum()
            )
            pairs["soil_moisture"].append((predicted, float(reference["soil_moisture"][60, 60])))
        for weights, value, valid in zip(
            reference["et_native_weights"][:],
            reference["et_native_value"][:],
            reference["et_native_valid"][:],
        ):
            if valid and np.isfinite(predictions["et"][weights > 0]).all():
                predicted = float((np.nan_to_num(predictions["et"]) * weights).sum())
                pairs["et"].append((predicted, float(value)))
        details.append(dict(window_id=row["window_id"], scenes=reference.attrs["scenes"]))
        print(f"Checked {row['window_id']}", flush=True)
    report = dict(
        time=args.date,
        drop_s2=args.drop_s2,
        paired_support=args.paired,
        reference_windows=len(details),
        interpretation="Teacher-product consistency only; SM uses window approximation",
        metrics={k: metrics(v) for k, v in pairs.items()},
        references=details,
    )
    gs.atomic_json(root / "reports" / f"{args.date}{suffix}_consistency.json", report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
