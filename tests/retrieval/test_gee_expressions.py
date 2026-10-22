"""Build real EE graphs offline using the API's published algorithm fixtures."""

from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase

from hydrostate.data.retrieval.composites import cover_composite, mean_composite, prepare_sensor
from hydrostate.data.retrieval.gee_labels import smap_label


class TestExpressions(ApiTestCase):
    def test_composite_graphs(self):
        image = ee.Image("example").set("system:time_start", ee.Date("2020-01-01").millis())
        for sensor in ("s1", "s2", "landsat"):
            prepared = prepare_sensor(ee, image, sensor)
            assert isinstance(prepared, ee.Image)
            assert prepared.serialize()
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
        with patch.object(ee.data, "computeValue", side_effect=responses):
            result = smap_label(ee, dict(lon=117.0, lat=27.0), cover, "2020-01-01", "2020-01-09")
        assert result["valid_dates"] == ["2020-01-01", "2020-01-05"]
        assert abs(result["value"] - 0.3) < 1e-8
        assert result["valid_days"] == 2
        assert result["grid"]["known"] == 1
