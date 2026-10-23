"""Build real EE graphs offline using the API's published algorithm fixtures."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase

from hydrostate.data.retrieval.composites import cover_composite, mean_composite, prepare_sensor
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.data.retrieval.gee_labels import fetch_products, smap_label

SM_CONFIG = dict(min_known_fraction=0.95, max_water_ice_fraction=0.05)


class TestExpressions(ApiTestCase):
    def test_composite_graphs(self):
        image = ee.Image("example").set("system:time_start", ee.Date("2020-01-01").millis())
        for sensor in ("s1", "s2", "landsat"):
            prepared = prepare_sensor(ee, image, sensor)
            assert isinstance(prepared, ee.Image)
            assert prepared.serialize()
        for group in INPUT_GROUPS.values():
            assert prepare_sensor(ee, image, group.sensor, group.bands).serialize()
        assert mean_composite(ee, ee.ImageCollection([image]), ["x"], "2020-01-01").serialize()
        assert cover_composite(
            ee, [], "2020-01-01", dict(water=0.35, snow_ice=0.6, land=0.85, land_max_water_ice=0.1)
        ).serialize()

    def test_smap_observation_graph_and_actual_dates(self):
        cover = ee.Image.constant([0, 0, 1, 1]).rename(["water", "snow_ice", "land", "known"])
        responses = [
            2,
            dict(crs="EPSG:6933", transform=[9000, 0, 0, 0, -9000, 0]),
            dict(
                features=[
                    dict(properties=dict(date="2020-01-01", soil_moisture_am=0.2)),
                    dict(properties=dict(date="2020-01-05", soil_moisture_am=0.4)),
                ]
            ),
            dict(known=100, water_ice=0, area=100),
            dict(known=100, water_ice=0, area=100),
            dict(DEM=20),
        ]
        with TemporaryDirectory() as directory:
            args = (
                ee,
                dict(lon=117.0, lat=27.0),
                cover,
                "2020-01-01",
                "2020-01-09",
                SM_CONFIG,
                Path(directory),
            )
            with patch.object(ee.data, "computeValue", side_effect=responses) as rpc:
                result = smap_label(*args)
                assert rpc.call_count == 6
            with patch.object(ee.data, "computeValue", side_effect=AssertionError("cache miss")):
                assert smap_label(*args) == result
        assert result["valid_dates"] == ["2020-01-01", "2020-01-05"]
        assert abs(result["value"] - 0.3) < 1e-8
        assert result["valid_days"] == 2
        assert result["grid"]["known"] == 1

    def test_smap_short_circuit_and_cover_cache_identity(self):
        cover = ee.Image.constant([0, 0, 1, 1]).rename(["water", "snow_ice", "land", "known"])
        base = [
            1,
            dict(crs="EPSG:6933", transform=[9000, 0, 0, 0, -9000, 0]),
            dict(features=[dict(properties=dict(date="2020-01-01", soil_moisture_am=0.2))]),
        ]
        bad, good = dict(known=90, water_ice=0, area=100), dict(known=100, water_ice=0, area=100)
        with TemporaryDirectory() as directory:

            def run(expression):
                return smap_label(
                    ee,
                    dict(lon=117.0, lat=27.0),
                    expression,
                    "2020-01-01",
                    "2020-01-09",
                    SM_CONFIG,
                    Path(directory),
                )

            with patch.object(ee.data, "computeValue", side_effect=[*base, bad]) as rpc:
                result = run(cover)
                assert result["skipped"] == "grid_quality" and rpc.call_count == 4
                assert "footprint" not in result and "terrain_std_m" not in result
            # New cover must recalculate both supports, while retaining SM metadata/observations.
            with patch.object(ee.data, "computeValue", side_effect=[good, bad]) as rpc:
                result = run(cover.multiply(1))
                assert result["skipped"] == "footprint_quality" and rpc.call_count == 2
                assert "terrain_std_m" not in result

    def test_nonland_tile_skips_smap_completely(self):
        with TemporaryDirectory() as directory:
            with patch.object(ee.data, "computeValue", side_effect=AssertionError("SMAP request")):
                pml, sm = fetch_products(
                    ee,
                    dict(target_time="2020-01-01", target_end="2020-01-09"),
                    None,
                    None,
                    SM_CONFIG,
                    Path(directory),
                    sm_enabled=False,
                )
        assert pml is None and sm == dict(valid_days=0, skipped="tile_land_fraction")
