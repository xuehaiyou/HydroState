"""Packed pixel transport preserves slot identity, masks and physical inputs."""

from pathlib import Path
from unittest.mock import patch

import ee
import numpy as np
from ee.apitestcase import ApiTestCase

from hydrostate.data.gee.products import BANDS, MODALITIES
from hydrostate.data.retrieval import collector
from hydrostate.data.retrieval.composites import QUALITY
from hydrostate.data.retrieval.contract import RetrievalSchema, intervals


class TestPackedInputs(ApiTestCase):
    def test_slots_bands_quality_and_missing_modalities(self):
        sample = dict(
            target_time="2020-01-17",
            target_end="2020-01-25",
            crs="EPSG:32650",
            transform=[10, 0, 0, 0, -10, 0],
        )
        slots, schema = intervals(sample["target_time"]), RetrievalSchema()
        catalogs, norm = {}, {}
        present = dict(s1=[0, 1, 4], s2=[], landsat=[1, 4])
        for sensor in BANDS:
            catalogs[sensor] = [
                dict(
                    asset_id=f"{sensor}_{slot}",
                    **{
                        "system:time_start": collector.epoch(slots[slot][0]) * 1000,
                        "relativeOrbitNumber_start": 1,
                        "orbitProperties_pass": "ASCENDING",
                    },
                )
                for slot in present[sensor]
            ]
            norm[MODALITIES[sensor]] = {
                band.lower()
                if sensor == "s1"
                else ("B" + band[1:].zfill(2) if band != "B8A" else band): dict(mean=0, std=1)
                for band in BANDS[sensor]
            }
        # A different radar orbit must remain excluded despite packing.
        catalogs["s1"].append(
            dict(catalogs["s1"][0], asset_id="wrong_orbit", relativeOrbitNumber_start=2)
        )
        requests = []

        def composite(ee, sensor, ids, start):
            return ee.Image.constant([0] * (len(BANDS[sensor]) + len(QUALITY))).rename(
                [*BANDS[sensor], *QUALITY]
            )

        def pixels(ee, expression, names, crs, transform, width, height):
            assert expression.serialize() and crs == sample["crs"]
            requests.append(names)
            raw = np.zeros((height, width), dtype=[(name, "f4") for name in names])
            for name in names:
                slot, band = name.split("_", 1)
                raw[name] = int(slot[1:]) + 1
                if band == "days":
                    raw[name][0, 0] = 0
            return raw

        with (
            patch.object(
                collector, "catalog", side_effect=lambda ee, cache, sensor, *a: catalogs[sensor]
            ),
            patch.object(collector, "sensor_composite", side_effect=composite),
            patch.object(collector, "compute_pixels", side_effect=pixels),
        ):
            arrays, scenes, eligible = collector.input_arrays(
                ee, sample, Path("unused"), norm, dict(min_input_fraction=0.5, min_active_slots=2)
            )
        assert eligible and len(requests) == 2
        assert all(len(names) == len(set(names)) for names in requests)
        for sensor, populated in present.items():
            for slot in range(schema.time_steps):
                assert scenes[f"{sensor}_{slot}"] == (
                    [f"{sensor}_{slot}"] if slot in populated else []
                )
                if slot not in populated:
                    assert not arrays[sensor][slot].any()
                    assert not arrays[f"{sensor}_valid"][slot].any()
                    continue
                assert arrays[f"{sensor}_quality"][slot, 0, 1, 1] == slot + 1
                assert not arrays[sensor][slot, :, 0, 0].any()
                expected = (slot + 1) / (4 if sensor == "s1" else 2) + 0.5
                assert arrays[sensor][slot, 0, 1, 1] == np.float16(expected)
                assert (
                    arrays["observation_time"][slot, list(BANDS).index(sensor)]
                    == (collector.epoch(slots[slot][0]) + collector.epoch(slots[slot][1])) // 2
                )
