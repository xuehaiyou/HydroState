"""Station ET versus native PML land ET, independently of footprint training approval."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from affine import Affine
from pyproj import Geod, Transformer

from hydrostate.data.gee.contracts import atomic_json, fingerprint, map_point
from hydrostate.data.gee.products import collection
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.labels import native_geometry
from hydrostate.data.retrieval.specs import LABEL_SPECS, SOURCES

from .archives import sha256
from .comparison import metrics
from .temporal import epoch


def period_end(start):
    """PML's nominal MODIS calendar; the final annual period is shorter than eight days."""
    day = date.fromisoformat(start)
    return min(day + timedelta(days=8), date(day.year + 1, 1, 1)).isoformat()


def native_cell(lon, lat, projection):
    xy = Transformer.from_crs(4326, projection["crs"], always_xy=True).transform(lon, lat)
    return tuple(np.floor(map_point(~Affine(*projection["transform"]), *xy)).astype(int))


class LocalPML:
    """Only reuse a valid stored value for the identical native cell, version and period."""

    def __init__(self, manifest):
        self.root = Path(manifest).parent
        self.table = pd.read_parquet(manifest)

    def find(self, record, identity):
        rows = self.table[
            self.table.target_time.eq(record["target_time"])
            & self.table.target_end.eq(record["target_end"])
        ]
        if rows.empty:
            return None
        # Any complete cell inside a 1.2 km input tile must be within 2 km of its centre.
        _, _, distance = Geod(ellps="WGS84").inv(
            rows.lon.to_numpy(),
            rows.lat.to_numpy(),
            np.full(len(rows), record["lon"]),
            np.full(len(rows), record["lat"]),
        )
        for row in rows[distance < 2000].itertuples():
            path = self.root / row.label_path
            attrs = json.loads((path / ".zattrs").read_text())
            if attrs.get("specifications", {}).get("labels", {}).get("et_land") != asdict(
                LABEL_SPECS["et_land"]
            ):
                continue
            scene = attrs["label_sources"].get("pml")
            if (
                not scene
                or not any(scene["asset_id"].startswith(s.collection + "/") for s in SOURCES["pml"])
                or scene["system:time_start"] != epoch(record["target_time"]) * 1000
            ):
                continue
            projection_path = (
                self.root / "cache/pml_projection" / (fingerprint(scene["asset_id"]) + ".json")
            )
            if not projection_path.exists():
                continue
            projection = json.loads(projection_path.read_text())
            cell = native_cell(record["lon"], record["lat"], projection)
            sample = {k: attrs["sample"][k] for k in ("crs", "transform", "lon", "lat")}
            _, centres, _ = native_geometry(json.dumps([sample, projection], sort_keys=True))
            to_lonlat = Transformer.from_crs(sample["crs"], 4326, always_xy=True)
            labels = zarr.open_group(str(path), mode="r")
            for i, (y, x) in enumerate(centres):
                lon, lat = to_lonlat.transform(
                    *map_point(Affine(*sample["transform"]), x + 0.5, y + 0.5)
                )
                if native_cell(lon, lat, projection) != cell:
                    continue
                j = int(row.shard_index)
                if labels["et_land_native_valid"][j, i]:
                    return dict(
                        identity=identity,
                        asset_id=scene["asset_id"],
                        projection=projection,
                        start=record["target_time"],
                        end=record["target_end"],
                        ET=float(labels["et_land_native_value"][j, i]),
                        Ew=None,
                        response_source="local_product_native_value",
                        reused_metadata=str(path),
                        time_support="nominal_UTC_MODIS_period; annual_end_truncated",
                    )
        return None


def pml_observation(ee, record, cache, *, local_products=None):
    spec = LABEL_SPECS["et_land"]
    identity = dict(
        version="site_pml_point_v1",
        sources=[asdict(s) for s in SOURCES["pml"]],
        scale=spec.scale,
        offset=spec.offset,
        lon=record["lon"],
        lat=record["lat"],
        start=record["target_time"],
        end=record["target_end"],
    )

    def fetch():
        if local_products is not None:
            existing = local_products.find(record, identity)
            if existing is not None:
                return existing
        images = collection(ee, "pml").filterDate(identity["start"], identity["end"])
        count = images.size().getInfo()
        result = dict(identity=identity, response_source="GEE_native_point_query")
        if count == 0:
            return dict(result, reason="no_product_image", ET=None, Ew=None)
        if count != 1:
            raise ValueError("More than one PML image in a station period")
        image = images.first()
        meta = image.toDictionary(["asset_id", "system:time_start", "system:time_end"]).getInfo()
        start = pd.Timestamp(meta["system:time_start"], unit="ms", tz="UTC").date().isoformat()
        end = period_end(start)
        if start != identity["start"] or end != identity["end"]:
            raise ValueError("PML nominal calendar does not match the station UTC window")
        if meta.get("system:time_end") is not None and meta["system:time_end"] != epoch(end) * 1000:
            raise ValueError("PML declared end disagrees with nominal period")
        grid = image.select(spec.band).projection()
        values = (
            image.select(["ET", "Ew"])
            .multiply(spec.scale)
            .add(spec.offset)
            .reduceRegion(
                ee.Reducer.first(),
                ee.Geometry.Point([record["lon"], record["lat"]]),
                crs=grid,
                maxPixels=8,
            )
            .getInfo()
        )
        return dict(
            result,
            **meta,
            start=start,
            end=end,
            ET=values.get("ET"),
            Ew=values.get("Ew"),
            projection=grid.getInfo(),
            time_support="nominal_UTC_MODIS_period; annual_end_truncated",
        )

    value = cached_json(Path(cache) / f"{fingerprint(identity)}.json", fetch)
    if value["identity"] != identity:
        raise ValueError("PML point cache identity mismatch")
    return value


def pair_et(record, product):
    if record.get("split") != "train" or not record["et"]["valid"]:
        raise ValueError("ET diagnostics require temporally qualified training windows")
    if product.get("start") is not None and (
        product["start"] != record["target_time"] or product["end"] != record["target_end"]
    ):
        raise ValueError("Station and product temporal supports differ")
    value = product.get("ET")
    valid = value is not None and np.isfinite(value) and value >= 0
    corrected = record.get("et_corrected") or {}
    return dict(
        site_id=record["site_id"],
        group_id=record["group_id"],
        split=record["split"],
        target_time=record["target_time"],
        target_end=record["target_end"],
        lon=record["lon"],
        lat=record["lat"],
        frequency_bin=record["frequency_bin"],
        unit="mm/day",
        station_et=record["et"]["value"],
        site_value=record["et"]["value"] if valid else None,
        product_value=float(value) if valid else None,
        et_corrected=corrected.get("value") if corrected.get("valid") else None,
        station_quality=record["et"],
        source_file=record["source_file"],
        source_time_basis=record["time_basis"],
        utc_offset_hours=record["utc_offset_hours"],
        label_time_basis="UTC; duration-weighted valid intervals; missing not zero-filled",
        product=product,
        support="unresolved_tower_footprint_vs_native_PML_pixel",
        supervision_ready=False,
        reason=None if valid else product.get("reason", "PML_land_ET_missing_or_invalid"),
    )


def compare_et_points(ee, selection, output, *, product_manifest=None, workers=4):
    if not 1 <= workers <= 4:
        raise ValueError("Use one bounded pool of at most four GEE workers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = json.loads(Path(selection).read_text())
    local = LocalPML(product_manifest) if product_manifest else None

    def compare(record):
        return pair_et(record, pml_observation(ee, record, output / "cache", local_products=local))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(compare, points["selected"]))
    report = dict(
        status="diagnostic_only_not_training_acceptance_or_model_accuracy",
        selection_sha256=sha256(selection),
        selected=len(rows),
        rows=rows,
        overall=metrics(rows),
        per_site={
            s: metrics([r for r in rows if r["group_id"] == s])
            for s in sorted({r["group_id"] for r in rows})
        },
    )
    atomic_json(output / "et_pml_pairs.json", report)
    pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    ).to_csv(output / "et_pml_pairs.csv", index=False)
    if report["overall"]["windows"]:
        plot_et_comparison(report, output)
    return report


def plot_et_comparison(report, output):
    """One quantitative panel, all valid pairs, equal axes; no product-agreement selection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["rows"] if r["product_value"] is not None]
    if not rows:
        raise ValueError("No valid ET product pairs")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 6.2))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.20, top=0.78)
    limit = max(1, np.ceil(max(max(r["site_value"], r["product_value"]) for r in rows)))
    ax.plot([0, limit], [0, limit], color="0.65", linestyle="--", linewidth=1, zorder=1)
    colors, markers = ["#555555", "#0072B2", "#009E73", "#D55E00"], ["o", "s", "^", "D"]
    for i, site in enumerate(sorted({r["group_id"] for r in rows})):
        group = [r for r in rows if r["group_id"] == site]
        ax.scatter(
            [r["product_value"] for r in group],
            [r["site_value"] for r in group],
            s=35,
            color=colors[i % 4],
            marker=markers[i % 4],
            alpha=0.85,
            edgecolors="none",
            label=site.removeprefix("flux:"),
        )
    # A small shared margin keeps valid near-zero markers fully visible.
    margin = 0.03 * limit
    ax.set(
        xlim=(-margin, limit + margin),
        ylim=(-margin, limit + margin),
        xlabel="PML land ET (mm/day)",
        ylabel="Station ET from LE (mm/day)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, fontsize=8, title="Station", title_fontsize=8)
    fig.suptitle("Station–PML ET consistency pilot", y=0.97, fontsize=12)
    m = report["overall"]
    r2 = "undefined" if m["r2"] is None else f"{m['r2']:.2f}"
    fig.text(
        0.14,
        0.88,
        f"{m['windows']} windows · {m['sites']} stations\n"
        f"RMSE {m['rmse']:.3f} · Bias {m['bias']:+.3f} · R² {r2}",
        fontsize=9,
        va="top",
    )
    fig.text(
        0.14,
        0.03,
        "Standard LE; matched nominal UTC periods.\n"
        "Tower footprint vs. native pixel; spatial match unresolved.",
        fontsize=8,
        va="bottom",
    )
    output = Path(output)
    fig.savefig(output / "et_pml_scatter.png", dpi=600)
    fig.savefig(output / "et_pml_scatter.pdf")
    fig.savefig(output / "et_pml_scatter.svg")
    plt.close(fig)
