"""Conservative per-checkpoint exclusion against actual selected training windows."""

import pandas as pd

from hydrostate.data.site.splits import CircleIndex, tile_circle
from hydrostate.registry import DATASETS


def training_overlap(cfg, plan, margin_m=35000):
    if margin_m < 35000:
        raise ValueError("Use at least the existing 35 km coarse-support exclusion margin")
    definition = dict(cfg.train_dataloader.dataset)
    if definition.get("split") != "train":
        raise ValueError("Expected checkpoint's train dataset configuration")
    dataset = DATASETS.build(definition)
    parts = [dataset.product, *dataset.sites] if hasattr(dataset, "product") else [dataset]
    records = []
    for part in parts:
        if not part.table.split.eq("train").all():
            raise ValueError("Training audit contains non-training rows")
        for row in part.table.to_dict("records"):
            records.append(tile_circle(row))
    if not records:
        raise ValueError("Cannot declare independence without training support")
    index = CircleIndex(records)
    result = []
    for row in plan.to_dict("records"):
        support = tile_circle(row)
        hits = index.overlaps(support["lon"], support["lat"], support["radius_m"] + margin_m)
        result.append(
            dict(
                sample_id=row["sample_id"],
                training_overlap=bool(hits),
                overlapping_training_windows=len(hits),
            )
        )
    return pd.DataFrame(result), dict(
        training_windows=len(records),
        margin_m=margin_m,
        policy="all_dates_input_enclosing_circles_plus_coarse_margin",
    )
