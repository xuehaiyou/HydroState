"""Physical support, missing labels and the user-selected pure-pixel assumption."""

import numpy as np
import pandas as pd
from shapely.geometry import box

from hydrostate.data.manual_cover.labels import area_fractions, point_fractions
from hydrostate.data.manual_cover.splits import assign


def test_point_onehot_dedup_conflict_and_absence():
    points = pd.DataFrame(
        dict(output_row=[0, 0, 1, 1, 2], output_col=[0] * 5, cover_class=[0, 0, 1, 2, 2])
    )
    t = point_fractions(points)
    assert t["water"][0, 0, 0] == 1
    assert t["point_count"][0, 0, 0] == 2
    assert t["water_valid"].sum() == 2  # duplicates do not increase supervision
    assert t["point_conflict"][0, 1, 0] == 1
    assert not t["water_valid"][0, 1, 0]
    assert t["land"][0, 2, 0] == 1
    assert not t["water_valid"][0, 3, 0]
    assert not t["valid_area_fraction"].any()  # no measured area from points


def test_exact_area_and_invalid_coverage():
    # A vertical half pixel is water, with a hole in another pixel's annotation.
    valid = box(0, 0, 1200, 1200).difference(box(30, 1170, 45, 1200))
    water = box(0, 0, 15, 1200)
    t = area_fractions(water, valid, [10, 0, 0, 0, -10, 1200])
    assert t["water"][0, 0, 0] == 0.5
    assert t["water_valid"][0, 0, 0]
    assert t["valid_area_fraction"][0, 0, 1] == 0.5
    assert not t["water_valid"][0, 0, 1]
    assert t["water_valid"][0, 0, 2]  # valid zero is supervised
    assert not t["snow_ice_valid"].any() and not t["land_valid"].any()
    assert np.isclose(t["water"].sum() * 900, water.area)


def test_unannotated_area_is_not_nonwater():
    t = area_fractions(box(0, 0, 10, 10), box(0, 0, 10, 10), [10, 0, 0, 0, -10, 1200])
    assert not t["water_valid"].any()
    assert t["water"][0, -1, 0] == 1
    assert np.isclose(t["valid_area_fraction"][0, -1, 0], 1 / 9)


def test_region_all_dates_stay_together():
    rows = []
    glaciers = [
        ("SouthCascade", -121, 48),
        ("Sperry", -114, 48),
        ("Gulkana", -145, 63),
        ("Wolverine", -149, 60),
    ]
    from hydrostate.data.spatial import grid

    for name, lon, lat in glaciers + [(f"event{i}", -80 + 20 * i, 0) for i in range(7)]:
        crs, tr = grid(lon, lat)
        source = "glacier" if name in {g[0] for g in glaciers} else "sen1floods11"
        for day in range(2):
            rows.append(
                dict(
                    sample_id=f"{name}_{day}",
                    source=source,
                    region=("glacier:" if source == "glacier" else "flood:") + name,
                    lon=lon,
                    lat=lat,
                    crs=crs,
                    transform=tr,
                )
            )
    plan, gaps = assign(pd.DataFrame(rows))
    assert plan.groupby("region").split.nunique().eq(1).all()
    assert set(plan[plan.region.eq("glacier:SouthCascade")].split) == {"test"}
    assert gaps[gaps.different_split].support_gap_m.gt(35000).all()


def test_unified_manual_loss_density_and_missing_gradients():
    import torch

    from hydrostate.models.manual_cover_supervision import manual_cover_loss

    values = {
        c: torch.full((2, 1, 4, 4), 0.25, requires_grad=True) for c in ("water", "snow_ice", "land")
    }
    targets = {}
    for c in values:
        targets["manual_" + c] = torch.zeros((2, 1, 4, 4))
        targets["manual_" + c + "_valid"] = torch.zeros((2, 1, 4, 4), dtype=torch.bool)
        targets["manual_" + c + "_valid"][0, 0, 0, 0] = True
    targets["manual_water"][0, 0, 0, 0] = 1
    targets["manual_water"][1] = 0.5
    targets["manual_water_valid"][1] = True
    loss = manual_cover_loss(values, targets, weight=2)
    expected = 2 * ((0.75**2 + 0.25**2 + 0.25**2) / 3 + 0.25**2) / 2
    assert abs(loss.item() - expected) < 1e-6
    loss.backward()
    assert not values["land"].grad[1].any()
    assert not values["snow_ice"].grad[1].any()
    assert values["water"].grad[1].abs().sum() > 0


def test_flood_metrics_never_report_land_or_ice():
    from hydrostate.evaluation.manual_cover import Scores

    t = area_fractions(box(0, 0, 600, 1200), box(0, 0, 1200, 1200), [10, 0, 0, 0, -10, 1200])
    predicted = np.stack([t["water"][0], np.zeros((40, 40)), 1 - t["water"][0]])
    score = Scores("sen1floods11")
    score.add(predicted, t, np.ones((40, 40), dtype=bool))
    report = score.report()
    assert set(report["classes"]) == {"water"}
    assert set(report["fraction_metrics"]) == {"water"}
    assert report["fraction_metrics"]["water"]["rmse"] == 0
    assert report["classes"]["water"]["f1"] == 1
