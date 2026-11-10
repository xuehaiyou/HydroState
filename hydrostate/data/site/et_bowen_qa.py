"""Paired PML diagnostic by accepted spatial-quality stratum, never a selection gate."""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.gee.client import initialize
from hydrostate.data.gee.contracts import atomic_json

from .et_comparison import LocalPML, pml_observation


def agreement(observed, predicted):
    x, y = np.asarray(observed, float), np.asarray(predicted, float)
    error = y - x
    den = float(((x - x.mean()) ** 2).sum())
    return dict(
        n=len(x),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.abs(error).mean()),
        bias=float(error.mean()),
        r2=float(1 - (error**2).sum() / den) if den else None,
        r=float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else None,
    )


def compare(root, product, previous_cache, output, *, per_stratum=12):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    rows = pd.read_parquet(root / "samples.parquet")
    rows = rows[rows.et_support.eq("static_210m") & rows.target_end.le("2025-01-01")].copy()
    rows["homogeneity_stratum"] = np.where(
        rows.homogeneous_fraction.lt(0.95), "90_to_95", "95_to_100"
    )
    # Randomize before taking at most two windows per site within each quality stratum.
    selected = (
        rows.sample(frac=1, random_state=20261110)
        .groupby(["homogeneity_stratum", "physical_group_id"], sort=False)
        .head(2)
        .groupby("homogeneity_stratum", sort=False)
        .head(per_stratum)
    )
    selected.to_parquet(output / "selection.parquet", index=False)
    cache = output / "cache"
    if previous_cache.exists():
        shutil.copytree(previous_cache, cache, dirs_exist_ok=True)
    ee = initialize("cropland-477906")
    local = LocalPML(product)

    def query(record):
        product = pml_observation(ee, record, cache, local_products=local)
        return dict(
            sample_id=record["sample_id"],
            group_id=record["physical_group_id"],
            target_time=record["target_time"],
            target_end=record["target_end"],
            homogeneous_fraction=record["homogeneous_fraction"],
            classification_valid_fraction=record["classification_valid_fraction"],
            homogeneity_stratum=record["homogeneity_stratum"],
            uncorrected=record["et_uncorrected"],
            corrected=record["et_corrected"],
            pml=product.get("ET"),
            product=product,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(query, selected.to_dict("records")))
    atomic_json(output / "pairs.json", dict(rows=pairs))
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "product"} for r in pairs])
    valid = np.isfinite(flat[["uncorrected", "corrected", "pml"]].to_numpy(dtype=float)).all(axis=1)
    flat["common_valid"] = valid
    flat.to_csv(output / "source_data.csv", index=False)
    data = flat[valid]
    stats = {}
    for name, frame in [("all", data), *list(data.groupby("homogeneity_stratum"))]:
        if len(frame):
            stats[name] = dict(
                sites=int(frame.group_id.nunique()),
                before=agreement(frame.uncorrected, frame.pml),
                after=agreement(frame.corrected, frame.pml),
            )
    result = dict(
        status="completed",
        selected=len(flat),
        paired=len(data),
        by_stratum=stats,
        purpose="preparation QA only, no label/threshold/split selection from product agreement",
        support="same valid half-hours before/after; native PML tower cell, not footprint matched",
    )
    atomic_json(output / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "product", "previous-cache", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.root, a.product, a.previous_cache, a.output)), flush=True)


if __name__ == "__main__":
    main()
