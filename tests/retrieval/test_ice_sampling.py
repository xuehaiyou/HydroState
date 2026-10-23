"""Ice proposals retain their dates, bypass JRC gaps and share bounded workers."""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from unittest.mock import patch

import ee
import numpy as np
import pytest
from ee.apitestcase import ApiTestCase

from hydrostate.data.retrieval import collector
from hydrostate.data.retrieval.ice_candidates import IceCandidateSource
from hydrostate.data.retrieval.sampling import SamplingLedger, location_key
from hydrostate.data.retrieval.scheduling import SamplePlanner, completed_samples


def row(index, group=0, **extra):
    return dict(
        crs="EPSG:32650",
        transform=[10, 0, index * 2000, 0, -10, 0],
        sampling_group=group,
        frequency_bin=0,
        **extra,
    )


def test_ice_date_cannot_be_changed_to_improve_balance():
    fixed = row(0, 7, candidate_target_time="2020-01-01")
    ledger = SamplingLedger([dict(row(10), target_time="2020-01-01")])
    planner = SamplePlanner([fixed], ["2020-01-01", "2021-03-01"], ledger, random.Random(0), set())
    assert planner.next(ledger, SamplingLedger(), {0: 1, 7: 1})["target_time"] == "2020-01-01"
    planner = SamplePlanner([fixed], ["2021-03-01"], ledger, random.Random(0), set())
    with pytest.raises(ValueError, match="calendar"):
        planner.next(ledger, SamplingLedger(), {0: 1, 7: 1})


def test_same_location_date_cannot_fill_two_groups():
    ledger = SamplingLedger()
    planner = SamplePlanner(
        [row(0), row(0, 7, candidate_target_time="2020-01-01")],
        ["2020-01-01"],
        ledger,
        random.Random(0),
        set(),
    )
    first = planner.next(ledger, SamplingLedger(), {0: 1, 7: 1})
    ledger.add(first)
    assert planner.next(ledger, SamplingLedger(), {0: 1, 7: 1}) is None
    assert len(ledger.dates[location_key(first)]) == 1


def test_slow_query_shares_four_workers_and_water_refill_waits_for_drain():
    lock, filled = Lock(), Event()
    active = peak = 0

    @contextmanager
    def operation():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 4:
                filled.set()
        try:
            yield
        finally:
            with lock:
                active -= 1

    class Ice:
        queried = False

        def next_query(self, *args):
            if not self.queried:
                self.queried = True
                return {}

        def fetch(self, query):
            with operation():
                assert filled.wait(5), "Ice query blocked the three available sample workers"
                return [row(50, 7, candidate_target_time="2020-01-01")]

        def observe(self, sample):
            pass

    def collect(sample):
        with operation():
            assert filled.wait(5)
            return sample

    refills = []

    def refill():
        assert active == 0, "Water metadata started while other requests were active"
        refills.append(True)
        return [row(i) for i in range(3, 6)] if len(refills) == 1 else None

    ledger = SamplingLedger()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            completed_samples(
                workers,
                [row(i) for i in range(3)],
                ledger,
                {0: 6, 7: 1},
                4,
                collect,
                dates=["2020-01-01"],
                rng=random.Random(0),
                tried=set(),
                ice_source=Ice(),
                next_candidates=refill,
            )
        )
    assert len(results) == 7 and peak == 4 and refills
    assert ledger.counts == {0: 6, 7: 1}


@pytest.mark.parametrize("fraction,valid", [(0.49, True), (1.0, False)])
def test_ineligible_ice_centre_rejects_before_downloading_inputs(
    tmp_path, monkeypatch, fraction, valid
):
    monkeypatch.setattr(collector, "catalog", lambda *a: [])
    monkeypatch.setattr(collector, "fetch_cover", lambda *a: (None, {}))
    snow = np.ones((1, 40, 40), dtype=np.float32)
    mask = np.ones_like(snow, dtype=bool)
    snow[0, 20, 20], mask[0, 20, 20] = fraction, valid
    monkeypatch.setattr(
        collector,
        "cover_targets",
        lambda *a: dict(water_valid=np.ones_like(mask), snow_ice=snow, snow_ice_valid=mask),
    )
    monkeypatch.setattr(
        collector, "input_arrays", lambda *a: pytest.fail("expensive inputs fetched")
    )
    sample = row(0, 7, target_time="2020-01-01", candidate_target_time="2020-01-01")
    cfg = dict(min_known_fraction=0.95, min_cover_pixels=1, min_center_ice_fraction=0.5)
    assert collector.collect_sample(None, sample, tmp_path, cfg, {}, {}) is None
    rejection = json.loads(next((tmp_path / "rejected").glob("*.json")).read_text())
    assert rejection["reason"] == "insufficient target-window centre ice"


class TestIceEE(ApiTestCase):
    def test_query_graph_missing_jrc_and_cache_including_empty(self):
        with TemporaryDirectory() as directory:
            cfg = dict(
                seed=43,
                candidate_seed=42,
                candidate_block_degrees=2,
                candidate_dir=directory,
                split_margin_m=35000,
                split_seed=42,
                climate_asset="example",
                ice_confidence=0.6,
                ice_candidates_per_query=8,
            )
            source = IceCandidateSource(ee, cfg, ["2020-01-01"])
            feature = dict(
                geometry=dict(type="Point", coordinates=[117, 27]),
                properties=dict(ice=1, jrc_observed=0, occurrence=0, climate_class=0),
            )
            for offset, features in enumerate(([feature], [])):
                request = dict(
                    block=[116, 26, 2],
                    bounds=[116, 26, 116.5, 26.5],
                    target_time=f"2020-01-{1 + offset * 8:02d}",
                )
                with patch.object(
                    ee.data, "computeValue", return_value=dict(features=features)
                ) as rpc:
                    result = source.fetch(request)
                    assert rpc.call_count == 1
                with patch.object(
                    ee.data, "computeValue", side_effect=AssertionError("cache miss")
                ):
                    assert source.fetch(request) == result
                if result:
                    assert result[0]["frequency_bin"] == -1
                    assert result[0]["sampling_group"] == 7
                    assert result[0]["candidate_target_time"] == request["target_time"]
                    assert not result[0]["jrc_observed"]
            assert source.stats["cache_hits"] == 2
            assert len(list((Path(directory) / "dw_ice_v1").glob("*.json"))) == 2
