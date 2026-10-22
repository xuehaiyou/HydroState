import numpy as np
import pytest
import torch

from hydrostate.data.retrieval.contract import TASKS, intervals, product_end
from hydrostate.data.retrieval.labels import build_targets, cover_targets, sm_eligible
from hydrostate.data.retrieval.sampling import SamplingLedger, frequency_bin, quotas
from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.models.water_soil_coupling import WaterSoilCoupling


def test_seven_boundaries_and_calendar():
    assert [frequency_bin(v) for v in (0, 0.1, 19, 20, 40, 60, 80, 99, 100)] == [
        0,
        1,
        1,
        2,
        3,
        4,
        5,
        5,
        6,
    ]
    assert frequency_bin(float("nan")) is None
    assert max(quotas(20000).values()) - min(quotas(20000).values()) == 1
    slots = intervals("2020-01-01")
    assert len(slots) == 5 and slots[2] == ("2020-01-01", "2020-01-09")
    assert slots[0][0] == "2019-12-16"
    assert product_end("2020-12-26") == "2021-01-01"


def test_dates_overlap_and_location_cap():
    sample = dict(crs="EPSG:32650", transform=[10, 0, 10000, 0, -10, 10000], frequency_bin=1)
    ledger = SamplingLedger()
    for day in ("2020-01-01", "2020-04-01", "2021-07-01", "2022-10-01"):
        assert ledger.allows(sample, day)
        ledger.add(dict(sample, target_time=day))
    assert not ledger.allows(sample, "2023-01-01")
    nearby = dict(sample, transform=[10, 0, 10600, 0, -10, 10000])
    assert not ledger.allows(nearby, "2020-01-20")
    assert ledger.allows(nearby, "2020-02-10")


def test_unknown_cover_and_smap_are_not_land():
    raw = {
        name: np.zeros((120, 120), np.float32)
        for name in (
            "water",
            "snow_ice",
            "land",
            "known",
            "days",
            "first_day",
            "last_day",
            "observed_days",
            "unknown_fraction",
        )
    }
    raw["land"][:] = raw["known"][:] = raw["days"][:] = 1
    raw["known"][0, 0] = 0
    targets = cover_targets(raw)
    assert not targets["land_valid"][0, 0, 0]
    assert targets["land_valid"][0, 0, 1]
    clean = dict(
        value=0.2,
        valid_days=2,
        grid=dict(known=1.0, water_ice=0.01),
        footprint=dict(known=0.98, water_ice=0.01),
    )
    assert sm_eligible(clean)
    clean["footprint"]["known"] = 0.96  # .01 water + .04 unknown fails strict < .05.
    assert not sm_eligible(clean)


def test_weak_sm_mean_masks_and_gradient():
    shape = (1, 1, 2, 2)
    outputs = dict(
        water=torch.zeros(shape),
        snow_ice=torch.zeros(shape),
        soil_moisture=torch.tensor([[[[0.1, 0.9], [0.3, 0.5]]]], requires_grad=True),
    )
    samples = dict(
        water=torch.zeros(shape),
        water_valid=torch.ones(shape, dtype=torch.bool),
        snow_ice=torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        snow_ice_valid=torch.ones(shape, dtype=torch.bool),
        soil_moisture=torch.tensor([[[[0.3]]]]),
        soil_moisture_valid=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )
    coupling = WaterSoilCoupling(ice_task="snow_ice", min_window_land_fraction=0.1)
    support = coupling.prepare(outputs, samples, supervision=True)
    pred, target, valid = supervision_values(
        "soil_moisture", outputs["soil_moisture"], samples, support, TASKS
    )
    assert valid.item() and pred.item() == pytest.approx(0.3)
    pred.sum().backward()
    assert outputs["soil_moisture"].grad[0, 0, 0, 1] == 0
    assert outputs["soil_moisture"].grad[0, 0, 1, 1] == pytest.approx(1 / 3)
    # Prediction validity cannot depend on label-domain masks at evaluation time.
    samples["soil_moisture_domain_valid"] = torch.zeros(shape, dtype=torch.bool)
    assert coupling.prepare(outputs, samples).pixel_valid.all()


def test_pml_pure_branch_routing_and_mixed_rejection():
    # Native 300 m cells fully contained in a 1.2 km tile, known affine grids.
    sample = dict(crs="EPSG:32650", transform=[10, 0, 500000, 0, -10, 3000000], lon=117.0, lat=27.0)
    raw = {
        name: np.zeros((120, 120), np.float32)
        for name in (
            "water",
            "snow_ice",
            "land",
            "known",
            "days",
            "first_day",
            "last_day",
            "observed_days",
            "unknown_fraction",
        )
    }
    raw["known"][:] = raw["days"][:] = raw["land"][:] = 1
    raw["land"][:30] = 0
    raw["water"][:30] = 1
    raw["water"][:30, 30:60] = 0.5
    raw["land"][:30, 30:60] = 0.5
    raw["land"][30:60] = 0
    raw["snow_ice"][30:60] = 1
    pml = dict(
        ET=np.full((120, 120), 2.0),
        Ew=np.full((120, 120), 3.0),
        ET_valid=np.ones((120, 120)),
        Ew_valid=np.ones((120, 120)),
        projection=dict(crs=sample["crs"], transform=[300, 0, 500000, 0, -300, 3000000]),
    )
    config = dict(min_known_fraction=0.95, max_water_ice_fraction=0.05, min_tile_land_fraction=0.95)
    target = build_targets(sample, raw, pml, dict(valid_days=0), config)
    assert target["et_water_native_valid"].sum() == 3
    assert target["et_ice_native_valid"].sum() == 4
    assert target["et_land_native_valid"].sum() == 8
    assert not target["soil_moisture_valid"].any()
    assert target["water_valid"].all()  # Mixed ET rejection does not reject coverage.
