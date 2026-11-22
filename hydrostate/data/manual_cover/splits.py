"""Whole-region partitions and conservative exclusions for future training."""

import itertools

import numpy as np
import pandas as pd

from hydrostate.data.site.splits import GEOD, CircleIndex, Groups, tile_circle

MARGIN_M = 35000


def assign(plan):
    """Merge enclosing regional supports across sources; never split an event or glacier."""
    names = sorted(plan.region.unique())
    circles = []
    for name in names:
        frame = plan[plan.region.eq(name)]
        lon, lat = float(frame.lon.mean()), float(frame.lat.mean())
        distances = GEOD.inv(
            np.full(len(frame), lon),
            np.full(len(frame), lat),
            frame.lon.to_numpy(),
            frame.lat.to_numpy(),
        )[2]
        radius = float(max(distances)) + max(
            tile_circle(r)["radius_m"] for r in frame.to_dict("records")
        )
        circles.append(dict(lon=lon, lat=lat, radius_m=radius))
    groups = Groups(len(names))
    tree = CircleIndex(circles)
    for i, c in enumerate(circles):
        for j in tree.overlaps(c["lon"], c["lat"], c["radius_m"] + MARGIN_M):
            groups.join(i, j)
    membership = {n: names[groups.root(i)] for i, n in enumerate(names)}
    plan = plan.copy()
    plan["spatial_group"] = plan.region.map(membership)
    # These glacier regions are widely separated. Final audit verifies that fact.
    fixed = {
        "glacier:SouthCascade": "test",
        "glacier:Sperry": "val",
        "glacier:Gulkana": "train",
        "glacier:Wolverine": "train",
    }
    assignments = {}
    for region, split in fixed.items():
        group = membership[region]
        if group in assignments and assignments[group] != split:
            raise ValueError("Glacier holdouts overlap; revise region allocation explicitly")
        assignments[group] = split
    flood = plan[plan.source.eq("sen1floods11")]
    sizes = flood.groupby("spatial_group").size()
    free = sorted(set(sizes.index) - set(assignments))
    if len(free) > 13:
        raise ValueError("Unexpected event count; bounded exhaustive split needs review")
    best = None
    labels = ("train", "val", "test")
    for choice in itertools.product(labels, repeat=len(free)):
        candidate = {**assignments, **dict(zip(free, choice))}
        counts = np.array([sum(n for g, n in sizes.items() if candidate[g] == s) for s in labels])
        group_counts = [sum(candidate[g] == s for g in sizes.index) for s in labels]
        if min(group_counts) < 2:
            continue
        score = float(np.sum((counts / counts.sum() - [0.6, 0.2, 0.2]) ** 2))
        if best is None or score < best[0]:
            best = score, candidate
    if best is None:
        raise ValueError("Cannot allocate at least two flood regions per split")
    plan["split"] = plan.spatial_group.map(best[1])
    report = []
    for i, a in enumerate(names):
        for j in range(i):
            b = names[j]
            ca, cb = circles[i], circles[j]
            gap = GEOD.inv(ca["lon"], ca["lat"], cb["lon"], cb["lat"])[2]
            gap -= ca["radius_m"] + cb["radius_m"]
            different = best[1][membership[a]] != best[1][membership[b]]
            if different and gap <= MARGIN_M:
                raise ValueError("Cross-split regional support overlap")
            report.append(
                dict(region_a=a, region_b=b, support_gap_m=gap, different_split=different)
            )
    return plan, pd.DataFrame(report)


def exclusions(plan, references):
    """Return BOTH sides of holdout conflicts, bound to explicit published manifests.

    References contain sample_id, manifest, and lon/lat/radius_m. Callers include
    actual SMAP supports where available, and conservative station footprints.
    """
    if references.empty:
        raise ValueError("Cannot publish without an existing-training support audit")
    holdouts = plan[~plan.split.eq("train")]
    support = [
        dict(tile_circle(r), sample_id=r["sample_id"], split=r["split"])
        for r in holdouts.to_dict("records")
    ]
    tree = CircleIndex(support)
    rows, nearest = [], []
    for record in references.to_dict("records"):
        hits = tree.overlaps(record["lon"], record["lat"], record["radius_m"] + MARGIN_M)
        if hits:
            rows.append(
                dict(
                    record,
                    reason="manual_holdout_support_plus_35km",
                    heldout_sample_ids=";".join(support[i]["sample_id"] for i in hits),
                )
            )
    # Exact nearest reference support gap for every manual sample; vectorized geodesics.
    lon, lat, radii = (references[c].to_numpy() for c in ("lon", "lat", "radius_m"))
    for r in plan.to_dict("records"):
        c = tile_circle(r)
        d = GEOD.inv(np.full(len(lon), c["lon"]), np.full(len(lat), c["lat"]), lon, lat)[2]
        gap = d - radii - c["radius_m"]
        j = int(np.argmin(gap))
        nearest.append(
            dict(
                sample_id=r["sample_id"],
                nearest_training_gap_m=float(gap[j]),
                nearest_manifest=references.iloc[j].manifest,
                nearest_training_id=references.iloc[j].sample_id,
                existing_training_proximity=bool(gap[j] <= MARGIN_M),
            )
        )
    return pd.DataFrame(
        rows, columns=[*references.columns, "reason", "heldout_sample_ids"]
    ), pd.DataFrame(nearest)
