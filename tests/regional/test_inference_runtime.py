import numpy as np
import pytest
import torch

from hydrostate.data.retrieval.contract import TASKS
from hydrostate.inference.regional import RegionalMosaic
from hydrostate.inference.regional_reports import check_fields, display_values
from hydrostate.inference.regional_runtime import select_batch


def test_offset_overlap_is_cosine_weighted_and_missing_neighbor_cannot_poison(tmp_path):
    mosaic = RegionalMosaic(tmp_path, (40, 56), TASKS, 32)
    for col, water, rate in ((0, 0.5, 10.0), (48, 0.25, 2.0)):
        values = dict(
            water=water,
            snow_ice=0.0,
            land=1 - water,
            soil_moisture=0.3,
            et_water=rate,
            et_ice=1.0,
            et_land=3.0,
        )
        output = {k: torch.full((1, 40, 40), v) for k, v in values.items()}
        output["prediction_valid"] = torch.ones((1, 40, 40), dtype=torch.bool)
        output["soil_moisture_prediction_valid"] = output["prediction_valid"].clone()
        if col:
            output["prediction_valid"][0, 10, 10] = False
            output["soil_moisture_prediction_valid"][0, 10, 10] = False
            for task in TASKS:
                output[task][0, 10, 10] = np.nan
        mosaic.add(0, col, output)
    # Same ground pixel lies at different positions within the overlapping cores.
    y, x = 20, 24
    a, b = mosaic.kernel[y - 4, x - 4], mosaic.kernel[y - 4, x - 20]
    wf = mosaic.sums["water"][y, x] / mosaic.weights["water"][y, x]
    ew = mosaic.sums["et_water"][y, x] / mosaic.weights["et_water"][y, x]
    np.testing.assert_allclose(wf, (a * 0.5 + b * 0.25) / (a + b))
    np.testing.assert_allclose(wf * ew, (a * 0.5 * 10 + b * 0.25 * 2) / (a + b), rtol=1e-6)
    for task in TASKS:
        assert np.isfinite(mosaic.weights[task]).all()
        assert np.isfinite(mosaic.sums[task]).all()
    assert mosaic.weights["et_water"][10, 26] > 0
    assert mosaic.sums["et_water"][10, 26] / mosaic.weights["et_water"][10, 26] == pytest.approx(10)
    mosaic.close()


def test_physical_audit_and_display_mask_are_distinct():
    fields = {
        "water": np.array([0.01, 0.5]),
        "snow_ice": np.array([0.0, 0.0]),
        "land": np.array([0.99, 0.5]),
        "et_water": np.array([5.0, 6.0]),
    }
    check_fields(fields)
    displayed = display_values(fields["et_water"], [True, True], fields["water"])
    assert np.isnan(displayed[0]) and displayed[1] == 6
    np.testing.assert_array_equal(fields["et_water"], [5, 6])
    fields["land"][0] = 0.5
    with pytest.raises(ValueError, match="do not close"):
        check_fields(fields)


def test_batch_selection_uses_largest_tested_size_with_headroom():
    results = [
        dict(batch_size=b, state="ok", windows_per_second=s, memory_fraction=m)
        for b, s, m in ((2, 10, 0.3), (4, 20, 0.5), (8, 20.5, 0.7), (16, 25, 0.9))
    ]
    assert select_batch(results) == 8
    with pytest.raises(ValueError, match="headroom"):
        select_batch([dict(batch_size=2, state="oom")])
