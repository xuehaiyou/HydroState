"""Conservative geodesic support isolation, independent of label agreement or target date."""

import itertools
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Geod, Transformer

from hydrostate.data.gee.contracts import atomic_json, fingerprint, map_point
from hydrostate.data.retrieval.contract import VERSION as PRODUCT_VERSION

from .archives import sha256
from .pilot import site_grid

GEOD = Geod(ellps="WGS84")
SPLIT_VERSION = "site_support_split_v1"


@lru_cache(maxsize=256)
def transformer(source, target):
    return Transformer.from_crs(source, target, always_xy=True)


def xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    # Below the minimum WGS84 curvature radius: the spatial hash cannot prune real overlaps.
    return 6300000 * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class CircleIndex:
    """Spatial hash uses a lower Earth radius for safe candidate screening, including dateline."""

    def __init__(self, rows, cell_size=20000):
        self.rows = rows
        self.cell_size = cell_size
        self.max_radius = max((r["radius_m"] for r in rows), default=0)
        self.buckets = defaultdict(list)
        for i, row in enumerate(rows):
            key = tuple(np.floor(xyz(row["lon"], row["lat"]) / cell_size).astype(int))
            self.buckets[key].append(i)

    def overlaps(self, lon, lat, radius):
        centre = xyz(lon, lat)
        reach = radius + self.max_radius
        low = np.floor((centre - reach) / self.cell_size).astype(int)
        high = np.floor((centre + reach) / self.cell_size).astype(int)
        found = []
        for key in itertools.product(*(range(a, b + 1) for a, b in zip(low, high))):
            for i in self.buckets.get(key, []):
                row = self.rows[i]
                distance = GEOD.inv(lon, lat, row["lon"], row["lat"])[2]
                if distance <= radius + row["radius_m"]:
                    found.append(i)
        return sorted(found)


def tile_circle(row):
    transform = row["transform"]
    affine = Affine(*(json.loads(transform) if isinstance(transform, str) else transform))
    corners = np.array([map_point(affine, *p) for p in ((0, 0), (120, 0), (120, 120), (0, 120))])
    lon, lat = transformer(row["crs"], "EPSG:4326").transform(*corners.T)
    radius = max(GEOD.inv(row["lon"], row["lat"], x, y)[2] for x, y in zip(lon, lat)) + 2
    return dict(lon=row["lon"], lat=row["lat"], radius_m=radius, kind="input_tile_enclosing_circle")


def smap_circle(support):
    affine = Affine(*support["grid"]["transform"])
    col, row = support["cell"]
    project = transformer(support["grid"]["crs"], "EPSG:4326")
    lon, lat = project.transform(*(map_point(affine, col + 0.5, row + 0.5)))
    corners = [
        project.transform(*(map_point(affine, *p)))
        for p in ((col, row), (col + 1, row), (col + 1, row + 1), (col, row + 1))
    ]
    if support["footprint"] != "33km_diameter_disk_approx":
        raise ValueError("Unknown product SMAP support; cannot certify spatial independence")
    radius = max(16500, *(GEOD.inv(lon, lat, x, y)[2] for x, y in corners)) + 2
    return dict(lon=lon, lat=lat, radius_m=radius, kind="smap_cell_and_33km_enclosing_circle")


def product_support(manifest, output):
    """Current PML accepted cells are completely inside the tile; valid SMAP adds its support."""
    manifest, output = Path(manifest), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = sha256(manifest)
    dest = output / "product_support.json"
    if dest.exists():
        cached = json.loads(dest.read_text())
        if cached["manifest_sha256"] != identity:
            raise ValueError("Product manifest changed; rebuild spatial index in a new directory")
        return cached
    rows = []
    table = pd.read_parquet(manifest)
    if set(table.version) != {PRODUCT_VERSION} or table.sample_id.duplicated().any():
        raise ValueError("Unsupported or duplicate product manifest")
    for record in table.to_dict("records"):
        common = dict(sample_id=record["sample_id"], product_split=record["split"])
        rows.append(dict(tile_circle(record), **common))
        if record["valid_label_counts"]["soil_moisture_valid"]:
            path = manifest.parent / record["label_path"] / ".zattrs"
            attrs = json.loads(path.read_text())
            rows.append(dict(smap_circle(attrs["sm_quality"]["support"]), **common))
    result = dict(
        manifest=str(manifest.resolve()),
        manifest_sha256=identity,
        version=SPLIT_VERSION,
        circles=rows,
        rule="Conservative enclosing circles; PML complete native cells inside input tile",
    )
    atomic_json(dest, result)
    return result


class Groups:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def assign_splits(stations, products, *, seed=43, development_sites=()):
    """Same physical identity and intersecting station input supports cannot cross splits."""
    stations = stations.sort_values("location_id").reset_index(drop=True).copy()
    located = stations.dropna(subset=["lon", "lat"]).copy().reset_index(drop=True)
    rows = located.to_dict("records")
    circles = [tile_circle(dict(row, **site_grid(row["lon"], row["lat"]))) for row in rows]
    physical, spatial = Groups(len(rows)), Groups(len(rows))
    identities = {}
    neighbours = CircleIndex(circles, cell_size=2000)
    for i, row in enumerate(rows):
        identity = row["group_id"]
        if identity in identities:
            physical.join(i, identities[identity])
            spatial.join(i, identities[identity])
        identities[identity] = i
        for j in neighbours.overlaps(row["lon"], row["lat"], circles[i]["radius_m"]):
            spatial.join(i, j)
            if GEOD.inv(row["lon"], row["lat"], rows[j]["lon"], rows[j]["lat"])[2] <= 30:
                physical.join(i, j)
    product_tree = CircleIndex(products)
    hits = [product_tree.overlaps(r["lon"], r["lat"], c["radius_m"]) for r, c in zip(rows, circles)]
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[spatial.root(i)].append(i)
    assignments = {}
    for root, members in groups.items():
        key = min(rows[i]["location_id"] for i in members)
        draw = int(fingerprint([seed, key])[:8], 16) % 100
        split = "train" if draw < 70 else "validation" if draw < 85 else "test"
        reason = "seeded_group_assignment"
        if any(hits[i] for i in members):
            split, reason = "train", "product_overlap_requires_train"
        if any(rows[i]["site_id"] in development_sites for i in members):
            split, reason = "train", "previous_development_group"
        assignments[root] = (key, split, reason)
    output, overlap_rows = [], []
    physical_keys = {}
    for i, row in enumerate(rows):
        root = physical.root(i)
        physical_keys[root] = min(physical_keys.get(root, row["group_id"]), row["group_id"])
    for i, row in enumerate(rows):
        key, split, reason = assignments[spatial.root(i)]
        matched = sorted({products[j]["sample_id"] for j in hits[i]})
        output.append(
            dict(
                row,
                physical_group_id=physical_keys[physical.root(i)],
                spatial_group_id=key,
                split=split,
                split_reason=reason,
                product_overlap_count=len(matched),
            )
        )
        overlap_rows.extend(
            dict(
                location_id=row["location_id"],
                site_split=split,
                product_sample_id=products[j]["sample_id"],
                product_split=products[j]["product_split"],
                support_kind=products[j]["kind"],
            )
            for j in hits[i]
        )
    unresolved = stations[stations.lon.isna() | stations.lat.isna()].copy()
    unresolved["split"] = "unresolved"
    return pd.concat([pd.DataFrame(output), unresolved], ignore_index=True), overlap_rows


def split_index(index, manifest, output, *, seed=43):
    index, manifest, output = Path(index), Path(manifest), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version=SPLIT_VERSION,
        seed=seed,
        station_sha256=sha256(index / "stations.parquet"),
        product_sha256=sha256(manifest),
        colocated_m=30,
        development_sites=["ismn:WATERLINE:Kerasia"],
    )
    path = output / "split_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Split inputs or policy changed; use a new output directory")
    atomic_json(path, config)
    products = product_support(manifest, output)
    table, hits = assign_splits(
        pd.read_parquet(index / "stations.parquet"),
        products["circles"],
        seed=seed,
        development_sites=config["development_sites"],
    )
    table.to_parquet(output / "station_splits.parquet", index=False)
    overlaps = pd.DataFrame(
        hits,
        columns=["location_id", "site_split", "product_sample_id", "product_split", "support_kind"],
    )
    overlaps.to_parquet(output / "product_overlaps.parquet", index=False)
    excluded = set(overlaps.loc[overlaps.site_split.eq("train"), "product_sample_id"])
    frame = pd.read_parquet(manifest)
    keep = ~frame.sample_id.isin(excluded) | frame.split.eq("train")
    # This is an ID selection; original relative Zarr paths continue to resolve against source root.
    frame.loc[keep, ["sample_id", "split"]].to_parquet(
        output / "joint_product_selection.parquet", index=False
    )
    report = dict(
        complete=True,
        config=config,
        locations=len(table),
        splits=dict(Counter(table.split)),
        spatial_groups=table.spatial_group_id.nunique(),
        independent_eval_overlap_rows=int(overlaps.site_split.isin(["validation", "test"]).sum()),
        excluded_product_evaluation_samples=int((~keep).sum()),
        joint_product_selection_basis="All indexed train station supports, conservative",
        support_rule=products["rule"],
        unresolved_locations=int(table.split.eq("unresolved").sum()),
    )
    if report["independent_eval_overlap_rows"]:
        raise AssertionError("Station evaluation overlaps product support")
    atomic_json(output / "split_report.json", report)
    return report
