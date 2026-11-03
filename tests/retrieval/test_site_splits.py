import pandas as pd
import pytest

from hydrostate.data.site.splits import CircleIndex, assign_splits, smap_circle


def test_circle_query_dateline_and_latitude():
    rows = [dict(lon=179.99, lat=0, radius_m=2000), dict(lon=0, lat=89.99, radius_m=2000)]
    index = CircleIndex(rows)
    assert index.overlaps(-179.99, 0, 1000) == [0]
    assert index.overlaps(180, 89.99, 1000) == [1]
    assert index.overlaps(0, 0, 1000) == []


def test_valid_smap_support_uses_publication_cell_centre():
    circle = smap_circle(
        dict(
            grid=dict(crs="EPSG:4326", transform=[0.08, 0, -180, 0, -0.08, 85]),
            cell=[2250, 1062],
            footprint="33km_diameter_disk_approx",
        )
    )
    assert circle["lon"] == pytest.approx(0.04)
    assert circle["lat"] == pytest.approx(0)
    assert circle["radius_m"] == 16502


def test_station_train_overlap_allowed_eval_excluded_and_grouped():
    rows = [
        dict(
            location_id=f"{i:04d}",
            site_id=f"site{i}",
            group_id=f"g{i}",
            lon=float(i),
            lat=0.0,
            source="ismn",
        )
        for i in range(80)
    ]
    # A second source and nearby neighbouring station must share the split.
    rows += [
        dict(rows[0], location_id="copy", site_id="copy", lon=0.0001),
        dict(rows[0], location_id="neighbor", site_id="neighbor", group_id="neighbor", lon=0.005),
    ]
    products = [
        dict(
            lon=0.0,
            lat=0.0,
            radius_m=16502,
            sample_id="product_test",
            product_split="test",
            kind="smap",
        )
    ]
    table, hits = assign_splits(pd.DataFrame(rows), products)
    assert set(table.loc[table.lon.lt(0.01), "split"]) == {"train"}
    assert table.loc[table.location_id.eq("copy"), "physical_group_id"].item() == "g0"
    assert all(hit["site_split"] == "train" for hit in hits)
    assert {"train", "validation", "test"} == set(table.split)
    shuffled, _ = assign_splits(pd.DataFrame(rows).sample(frac=1, random_state=4), products)
    assert table[["location_id", "split"]].equals(shuffled[["location_id", "split"]])


def test_previously_developed_site_group_cannot_be_heldout():
    rows = [dict(location_id="abc", site_id="developed", group_id="dev", lon=70.0, lat=30.0)]
    table, _ = assign_splits(pd.DataFrame(rows), [], development_sites=["developed"])
    assert table.split.item() == "train"
    assert table.split_reason.item() == "previous_development_group"
