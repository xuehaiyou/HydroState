"""Collect HLS reflectance for the immutable comparison windows, daily then eight-day mean."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.backends.pixels import compute_pixels
from hydrostate.data.gee.client import initialize

from .composites import QUALITY, mean_composite
from .contract import intervals

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
COLUMNS = ["sample_id", "target_time", "crs", "transform", "lat", "lon"]


def sidecar(root, sample_id):
    key = hashlib.sha256(sample_id.encode()).hexdigest()
    return Path(root) / key[:2] / (key + ".npz")


def collect(ee, root, row):
    path = sidecar(root, row.sample_id)
    if path.exists():
        return
    affine = json.loads(row.transform)
    region = ee.Geometry.Rectangle(
        [affine[2], affine[5] - 1200, affine[2] + 1200, affine[5]], row.crs, False
    )
    slots = intervals(str(row.target_time)[:10])
    collections = []
    for source, names in (
        ("HLSL30", ["B2", "B3", "B4", "B5", "B6", "B7"]),
        ("HLSS30", ["B2", "B3", "B4", "B8A", "B11", "B12"]),
    ):

        def prepare(im, names=names):
            x = im.select(names, BANDS)
            ok = im.select("Fmask").bitwiseAnd(15).eq(0).And(x.mask().reduce(ee.Reducer.min()))
            return x.updateMask(ok).copyProperties(im, ["system:time_start"])

        collections.append(
            ee.ImageCollection(f"NASA/HLS/{source}/v002")
            .filterBounds(region)
            .filterDate(slots[0][0], slots[-1][1])
            .map(prepare)
        )
    scenes = collections[0].merge(collections[1])
    images, keys = [], []
    for i, (start, end) in enumerate(slots):
        names = [f"t{i}_{name}" for name in BANDS + QUALITY]
        images.append(mean_composite(ee, scenes.filterDate(start, end), BANDS, start).rename(names))
        keys.extend(names)
    grid = [v * 3 if i in (0, 1, 3, 4) else v for i, v in enumerate(affine)]
    raw = compute_pixels(ee, ee.Image.cat(images), keys, row.crs, grid, 40, 40)
    values = np.stack([np.stack([raw[f"t{i}_{b}"] for b in BANDS]) for i in range(5)])
    quality = np.stack([np.stack([raw[f"t{i}_{b}"] for b in QUALITY]) for i in range(5)])
    if np.abs(values).max() > 10 or quality[:, 0].max() > 8:
        raise ValueError(f"Unexpected HLS units or coverage for {row.sample_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            hls=values.astype("float16"),
            quality=quality.astype("float16"),
            location=np.float32([row.lat, row.lon]),
        )
    tmp.replace(path)


def main():
    from mmengine.config import Config

    from hydrostate.registry import DATASETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    cfg = Config.fromfile(args.reference)
    tables = []
    for split, expected in (("train", 10912), ("val", 3681), ("test", 11168)):
        ds = DATASETS.build(cfg[split + "_dataloader"].dataset)
        if len(ds) != expected:
            raise ValueError(f"Changed {split} selection")
        tables.extend(v.table[COLUMNS] for v in (*ds.components, ds.manual))
    table = pd.concat(tables).drop_duplicates()
    if table.sample_id.duplicated().any():
        raise ValueError("Conflicting sample grids")
    args.root.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.root / "selection.parquet", index=False)
    ee = initialize(args.project)
    if args.pilot:
        table = table.iloc[:2]
    with ThreadPoolExecutor(4) as pool:
        for i, _ in enumerate(
            pool.map(lambda row: collect(ee, args.root, row), table.itertuples(index=False)), 1
        ):
            if i % 100 == 0:
                print(f"{i}/{len(table)}", flush=True)
    if not args.pilot:
        (args.root / "complete.json").write_text(json.dumps({"windows": len(table)}))


if __name__ == "__main__":
    main()
