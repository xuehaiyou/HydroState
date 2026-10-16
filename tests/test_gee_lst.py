"""LST selection must use real input scenes, actual times and the unchanged 2 K QA."""

import numpy as np
import pytest

from hydrostate.data.gee_lst import choose_observation, input_scene_pairs, order_candidates
from hydrostate.data.target_times import epoch_seconds, product_time_bounds


def scene(day, asset="a", slot=None):
    value = dict(asset_id=asset, time_ms=epoch_seconds(day + "T10:30:00Z") * 1000)
    if slot is not None:
        value.update(input_slot=slot, input_time_ms=value["time_ms"])
    return value


def values(uncertainty=2., valid=True):
    return dict(lst=np.full((1, 40, 40), 300., "float32"),
                lst_valid=np.full((1, 40, 40), valid),
                lst_uncertainty=np.full((1, 40, 40), uncertainty, "float32"))


def test_preserve_good_same_day_without_query_or_download():
    old = scene("2020-01-10")
    def forbidden(*args):
        raise AssertionError("A qualifying baseline label needs no GEE access")
    support = np.ones((1, 40, 40), bool)
    output, meta = choose_observation(dict(target_time="2020-01-10"), support, {},
                                     (values(), old), forbidden, forbidden)
    assert output["lst_valid"].all()
    assert output["lst_time"].tolist() == [old["time_ms"] // 1000] * 2
    assert meta["selection"] == "preserved_same_day"


def test_try_actual_quality_in_order_and_keep_only_selected_scene_pixels():
    candidates = [scene("2020-01-07", "far", 0), scene("2020-01-10", "same"),
                  scene("2020-01-09", "near", 1)]
    support = np.ones((1, 40, 40), bool)
    paired = {0: support.copy(), 1: support.copy()}
    paired[0][..., 0, 0] = False
    calls = []
    def fetch(item):
        calls.append(item["asset_id"])
        return values(2.01 if item["asset_id"] == "near" else 2.,
                      valid=item["asset_id"] != "same")
    result, meta = choose_observation(dict(target_time="2020-01-10"), support,
                                     paired, None, lambda: candidates, fetch)
    assert calls == ["same", "near", "far"]
    assert result["lst_valid"].sum() == 1599
    assert result["lst_time"][0] == epoch_seconds("2020-01-07T10:30:00Z")
    assert meta["selection"] == "paired_input"
    assert meta["scenes"]["lst"]["asset_id"] == "far"


def test_missing_is_masked_and_download_errors_are_not_silenced():
    support = np.ones((1, 40, 40), bool)
    def candidates():
        return [scene("2020-01-10")]
    args = (dict(target_time="2020-01-10"), support, {}, None, candidates)
    result, meta = choose_observation(*args, lambda s: values(2.01))
    assert not result["lst_valid"].any()
    assert not result["lst_time"].any()
    assert meta["selection"] == "missing"
    def failed(s):
        raise RuntimeError("network error")
    with pytest.raises(RuntimeError, match="network error"):
        choose_observation(*args, failed)


def test_pair_identity_and_timestamp_are_checked():
    sample = dict(target_time="2020-01-10", inputs={
        "landsat_2": dict(asset_id="LANDSAT/LC08/C02/T1/LC08_123045_20200109",
                          **{"system:time_start": 1234567890000}), "s1_0": None})
    pairs = input_scene_pairs(sample)
    assert list(pairs) == ["LANDSAT/LC08/C02/T1_L2/LC08_123045_20200109"]
    assert next(iter(pairs.values()))["input_slot"] == 2
    with pytest.raises(ValueError, match="paired"):
        order_candidates([scene("2020-01-09")], sample)
    wrong = scene("2020-01-09", slot=2)
    wrong["input_time_ms"] += 1000
    with pytest.raises(ValueError, match="times disagree"):
        order_candidates([wrong], sample)


def test_product_intervals_and_subday_timestamps():
    scenes = dict(dw=scene("2020-01-10"), sm=scene("2020-01-10"),
                  lai=scene("2020-12-29"), gpp=scene("2020-12-26"),
                  pml=dict(interval_start="2020-01-01", interval_end="2020-01-09"),
                  albedo=dict(interval_start="2020-01-02", interval_end_exclusive="2020-01-18"))
    assert product_time_bounds(scenes, "water")[0] == epoch_seconds("2020-01-10T10:30:00Z")
    assert product_time_bounds(scenes, "lai")[1] == epoch_seconds("2021-01-01")
    assert product_time_bounds(scenes, "gpp")[1] == epoch_seconds("2021-01-01")
    assert np.diff(product_time_bounds(scenes, "et")) == 8 * 86400
    assert np.diff(product_time_bounds(scenes, "albedo_bsa")) == 16 * 86400
