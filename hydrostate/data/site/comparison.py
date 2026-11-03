"""Traceable station/SMAP diagnostic pairing; no selection by product agreement."""

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.gee.products import collection
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.specs import SOURCES


def smap_observations(ee, record, cache):
    """Sample the actual GEE publication grid; the catalogue exposes no overpass-time band."""
    identity = dict(
        sources=[asdict(s) for s in SOURCES["sm"]],
        lon=record["lon"],
        lat=record["lat"],
        start=record["target_time"],
        end=record["target_end"],
    )

    def fetch():
        images = collection(ee, "sm").filterDate(identity["start"], identity["end"])
        size = images.size().getInfo()
        if not size:
            return dict(identity=identity, observations=[], projection=None)
        point = ee.Geometry.Point([identity["lon"], identity["lat"]])

        def sample(image):
            image = ee.Image(image)
            values = image.select(["soil_moisture_am", "retrieval_qual_flag_am"]).reduceRegion(
                ee.Reducer.first(),
                point,
                crs=image.select("soil_moisture_am").projection(),
                maxPixels=8,
            )
            return ee.Feature(None, values).set(
                dict(date=image.date().format("YYYY-MM-dd"), asset_id=image.get("asset_id"))
            )

        observations = ee.FeatureCollection(images.toList(size).map(sample)).getInfo()["features"]
        return dict(
            identity=identity,
            observations=[f["properties"] for f in observations],
            projection=images.first().select("soil_moisture_am").projection().getInfo(),
        )

    return cached_json(Path(cache) / f"{fingerprint(identity)}.json", fetch)


def pair_sm(record, products):
    # Product granules are daily UTC composites. Near the date line use the ideal
    # morning's UTC day, not its local solar date or the image's nominal noon timestamp.
    mornings = {}
    for item in record["sm"]["morning"]:
        ideal = item["utc"] - item["offset_seconds"]
        day = datetime.fromtimestamp(ideal, UTC).date().isoformat()
        mornings[day] = item
    pairs, excluded = [], []
    seen = set()
    for product in products["observations"]:
        day, value = product["date"], product.get("soil_moisture_am")
        if day in seen:
            raise ValueError("Multiple SMAP observations for one date require version resolution")
        seen.add(day)
        if value is None or not 0 <= value <= 1 or product.get("retrieval_qual_flag_am") != 0:
            excluded.append(dict(date=day, reason="SMAP_missing_or_QA", product=product))
        elif day not in mornings:
            excluded.append(dict(date=day, reason="no_common_station_morning", product=product))
        else:
            pairs.append(
                dict(
                    date=day,
                    site=mornings[day]["value"],
                    product=value,
                    site_utc=mornings[day]["utc"],
                    asset_id=product["asset_id"],
                )
            )
    for day in sorted(set(mornings) - seen):
        excluded.append(dict(date=day, reason="SMAP_unavailable"))
    return dict(
        site_id=record["site_id"],
        group_id=record["group_id"],
        target_time=record["target_time"],
        training_value=record["sm"]["value"],
        common_days=len(pairs),
        site_value=float(np.mean([r["site"] for r in pairs])) if pairs else None,
        product_value=float(np.mean([r["product"] for r in pairs])) if pairs else None,
        paired_days=pairs,
        excluded_days=excluded,
        source_file=record["source_file"],
        time_match="ideal_morning_UTC_date_approximation; exact_overpass_not_exposed",
        support="point_vs_native_SMAP_grid; representativeness_not_yet_accepted",
        projection=products["projection"],
        reason=None if pairs else "no_common_QA_valid_dates",
    )


def metrics(rows):
    valid = [r for r in rows if r["site_value"] is not None and r["product_value"] is not None]
    result = dict(
        windows=len(valid), sites=len({r["group_id"] for r in valid}), rmse=None, bias=None, r2=None
    )
    if valid:
        site = np.array([r["site_value"] for r in valid])
        error = np.array([r["product_value"] for r in valid]) - site
        result.update(rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()))
        denominator = np.sum((site - site.mean()) ** 2)
        if len(site) > 1 and denominator > 0:
            result["r2"] = float(1 - np.sum(error**2) / denominator)
    return result


def collect_sm_comparison(ee, manifest, output):
    root, output = Path(manifest).parent, Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    table = pd.read_parquet(manifest)
    if not table.split.isin(["train", "development"]).all():
        raise ValueError("Pilot diagnostics cannot tune rules against held-out stations")
    for _, sample in table.iterrows():
        group = zarr.open_group(str(root / sample.label_path), mode="r")
        for record in group.attrs["observation_records"]:
            if (record.get("sm") or {}).get("valid"):
                paired = pair_sm(record, smap_observations(ee, record, output / "cache"))
                rows.append(
                    dict(
                        paired, sample_id=sample.sample_id, frequency_bin=int(sample.frequency_bin)
                    )
                )
    report = dict(
        status="diagnostic_only_spatial_acceptance_pending",
        rows=rows,
        overall=metrics(rows),
        per_site={
            site: metrics([r for r in rows if r["group_id"] == site])
            for site in sorted({r["group_id"] for r in rows})
        },
    )
    atomic_json(output / "sm_smap_pairs.json", report)
    flat = [
        {
            k: r[k]
            for k in (
                "sample_id",
                "site_id",
                "group_id",
                "target_time",
                "frequency_bin",
                "training_value",
                "site_value",
                "product_value",
                "common_days",
                "reason",
            )
        }
        for r in rows
    ]
    pd.DataFrame(flat).to_csv(output / "sm_smap_pairs.csv", index=False)
    return report


def plot_sm_comparison(report, output):
    """One diagnostic panel: raw paired windows, equal axes, no inferred independent accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["rows"] if r["site_value"] is not None]
    if not rows:
        raise ValueError("No paired SM windows to plot; inspect saved rejection reasons")
    output = Path(output)
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
    fig, ax = plt.subplots(figsize=(5.1, 5.5))
    fig.subplots_adjust(left=0.17, right=0.94, bottom=0.17, top=0.82)
    limit = max(0.1, np.ceil(max(max(r["site_value"], r["product_value"]) for r in rows) * 10) / 10)
    ax.plot([0, limit], [0, limit], color="0.65", linewidth=1, linestyle="--", zorder=1)
    colors = ["#555555", "#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    groups = ["0%", "(0, 20)%", "[20, 40)%", "[40, 60)%", "[60, 80)%", "[80, 100)%", "100%"]
    for frequency in sorted({r["frequency_bin"] for r in rows}):
        selected = [r for r in rows if r["frequency_bin"] == frequency]
        ax.scatter(
            [r["product_value"] for r in selected],
            [r["site_value"] for r in selected],
            s=38,
            color=colors[frequency],
            alpha=0.8,
            edgecolors="none",
            label=groups[frequency],
        )
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="SMAP AM (m³/m³)", ylabel="Station AM (m³/m³)")
    ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Station–SMAP consistency pilot", y=0.96, fontsize=12)
    m = report["overall"]
    r2 = "undefined" if m["r2"] is None else f"{m['r2']:.2f}"
    fig.text(
        0.17,
        0.89,
        f"{m['windows']} windows · {m['sites']} {'site' if m['sites'] == 1 else 'sites'}\n"
        f"RMSE {m['rmse']:.3f} · Bias {m['bias']:+.3f} · R² {r2}",
        fontsize=9,
    )
    ax.legend(
        title="Centre water frequency",
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    fig.text(
        0.17,
        0.055,
        "UTC-date approximation; point vs. coarse pixel.\n"
        "Spatial representativeness remains to be accepted.",
        fontsize=8,
    )
    fig.savefig(output / "sm_smap_scatter.png", dpi=600)
    fig.savefig(output / "sm_smap_scatter.pdf")
    fig.savefig(output / "sm_smap_scatter.svg")
    plt.close(fig)
