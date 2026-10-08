"""Plot completed sample centres, coloured by dataset split (requires Cartopy)."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    df = pd.read_parquet(io.BytesIO(source_bytes))
    if df.empty or df.sample_id.duplicated().any():
        raise ValueError("Expected nonempty manifest with unique sample IDs")
    if not (np.isfinite(df[["lon", "lat"]]).all().all()
            and df.lon.between(-180, 180).all()
            and df.lat.between(-90, 90).all()):
        raise ValueError("Invalid geographic coordinates")
    styles = {
        "train": ("Training", "#347FA3"),
        "validation": ("Validation", "#E7A33E"),
        "test": ("Test", "#AC4C70"),
    }
    if df.split.isna().any() or not set(df.split).issubset(styles):
        raise ValueError("Unknown or missing split")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, 8.1), facecolor="white")
    ax = fig.add_axes([0.025, 0.16, 0.95, 0.70], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("#F4F8FA")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#E5E7E4", zorder=0)
    ax.coastlines(resolution="110m", color="#8C9699", linewidth=0.45)
    ax.gridlines(xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-60, 91, 30),
                 color="#ABB6BD", alpha=0.45, linewidth=0.45, linestyle=":")
    ax.spines["geo"].set_edgecolor("#ADB7BC")
    ax.spines["geo"].set_linewidth(0.7)
    # Deterministic mixed draw order avoids putting one entire split on top.
    ordered = df.sample(frac=1, random_state=20260923)
    ax.scatter(ordered.lon, ordered.lat, s=6,
               c=[styles[x][1] for x in ordered.split], alpha=0.72,
               linewidths=0, transform=ccrs.PlateCarree(), zorder=3)
    counts = {key: int((df.split == key).sum()) for key in styles}
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      color=color, label=f"{label}  {counts[key]:,}")
               for key, (label, color) in styles.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.09),
               ncol=3, frameon=False, columnspacing=3)
    dates = pd.to_datetime(df.target_time)
    fig.text(0.04, 0.935, "Spatial distribution of collected samples", fontsize=21,
             weight="bold", color="#243745")
    fig.text(0.04, 0.89,
             f"{len(df):,} completed samples  |  {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}",
             fontsize=12, color="#566873")
    fig.text(0.04, 0.045,
             "Each point represents one sample centre; overlapping points may obscure samples.",
             fontsize=9, color="#647580")
    fig.text(0.96, 0.022, "Robinson projection | Basemap: Natural Earth", ha="right",
             fontsize=8, color="#647580")
    stem = args.output / "sample_distribution"
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)
    df[["sample_id", "lon", "lat", "split", "frequency_bin", "target_time"]].to_csv(
        args.output / "sample_locations.csv", index=False)
    summary = {"source": str(args.source.resolve()),
               "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "n_samples": len(df), "split_counts": counts,
               "unique_locations": len(df[["lon", "lat"]].drop_duplicates()),
               "longitude_range": [float(df.lon.min()), float(df.lon.max())],
               "latitude_range": [float(df.lat.min()), float(df.lat.max())]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
