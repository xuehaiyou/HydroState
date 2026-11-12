import pytest

from hydrostate.data.site.product_baseline import reduce_pml_cells
from hydrostate.evaluation.site_baseline_scoring import agreement, common_comparison


def test_native_pml_support_never_fills_or_renormalizes_missing_cells():
    cells = [dict(weight=.25, ET=2., land_purity=.96),
             dict(weight=.75, ET=6., land_purity=1.)]
    assert reduce_pml_cells(cells, .95) == (5., None)
    cells[0]["ET"] = None
    assert reduce_pml_cells(cells, .95)[0] is None
    cells[0]["ET"] = 0.  # Valid zero evaporation is not missing.
    assert reduce_pml_cells(cells, .95) == (4.5, None)
    cells[0]["land_purity"] = .95
    assert reduce_pml_cells(cells, .95)[0] is None
    with pytest.raises(ValueError):
        reduce_pml_cells(cells[1:], .95)


def pair(key, target, product, eligible=True):
    return dict(key=key, task="sm", spatial_group_id="g", station_value=target,
                original_target=target + .01, product_value=product, eligible=eligible)


def prediction(row, value, valid=True):
    return dict(key=row["key"], prediction=value,
                original_target=row["original_target"], valid=valid)


def test_every_method_uses_exactly_the_same_observations_and_common_target():
    pairs = [pair("a", .1, .2), pair("b", .4, .3), pair("c", .8, .9)]
    models = dict(B=[prediction(r, r["station_value"]) for r in pairs],
                  C=[prediction(r, r["station_value"], r["key"] != "c") for r in pairs])
    result, selected = common_comparison(pairs, models)
    assert [r["key"] for r in selected] == ["a", "b"]
    assert result["sm"]["metrics"]["B"]["rmse"] == 0
    assert result["sm"]["metrics"]["product"]["rmse"] == pytest.approx(.1)
    assert all(m["n"] == 2 for m in result["sm"]["metrics"].values())
    assert result["et"]["n"] == 0


def test_scoring_rejects_label_drift_and_duplicates():
    row = pair("a", .1, .2)
    bad = prediction(row, .1)
    bad["original_target"] = .8
    with pytest.raises(ValueError, match="different station"):
        common_comparison([row], dict(B=[bad]))
    with pytest.raises(ValueError, match="Duplicate"):
        common_comparison([row, row], dict(B=[]))
    with pytest.raises(ValueError, match="Duplicate"):
        common_comparison([row], dict(B=[bad, bad]))


def test_product_missing_samples_do_not_enter_any_method():
    row = pair("a", .1, None, False)
    result, selected = common_comparison([row], dict(B=[prediction(row, .1)]))
    assert not selected
    assert result["sm"]["metrics"]["product"]["n"] == 0


def test_r2_is_not_squared_correlation():
    assert agreement([1, 2, 3], [11, 12, 13])["r2"] == -149
    assert agreement([1], [2])["r2"] is None
