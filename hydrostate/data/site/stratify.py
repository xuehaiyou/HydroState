"""Small cached point queries for water frequency, landcover and climate; no image downloads."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.gee.products import collection
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.specs import SOURCES

from .archives import sha256
from .pilot import observed_occurrence


def enrich_index(ee, stations_path, output, climate_asset, *, workers=4, batch_size=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        stations_sha256=sha256(stations_path),
        climate_asset=climate_asset,
        policy="jrc_metadata_valid_observations_v1",
        batch_size=batch_size,
    )
    config_path = output / "strata_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Point query inputs changed; use a new output directory")
    atomic_json(config_path, config)
    table = pd.read_parquet(stations_path)
    points = table.dropna(subset=["lon", "lat"])[["lon", "lat"]].drop_duplicates()
    positions = sorted(map(tuple, points.to_numpy().tolist()))
    image = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(-1, False)
        .addBands(
            ee.Image("JRC/GSW1_4/Metadata").select(["valid_obs", "detections"]).unmask(-1, False)
        )
        .addBands(ee.Image(climate_asset).select(0).rename("climate_class").unmask(0, False))
        .addBands(
            ee.ImageCollection("ESA/WorldCover/v200")
            .first()
            .select("Map")
            .rename("landcover")
            .unmask(0, False)
        )
    )

    def query(batch):
        features = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point(list(p)), dict(point_id=i)) for i, p in enumerate(batch)]
        )
        key = fingerprint([config, batch])
        fetched = cached_json(
            output / "cache" / f"{key}.json",
            lambda: image.reduceRegions(
                collection=features, reducer=ee.Reducer.first(), scale=30
            ).getInfo()["features"],
        )
        if len(fetched) != len(batch):
            raise ValueError("Point query lost station positions")
        rows = []
        for feature in fetched:
            values = feature["properties"]
            lon, lat = batch[int(values.pop("point_id"))]
            row = dict(values, lon=lon, lat=lat, frequency_bin=-1, strata_status="unresolved")
            try:
                occurrence = observed_occurrence(values)
                frequency = (
                    0 if occurrence == 0 else 6 if occurrence == 100 else int(occurrence // 20) + 1
                )
                row.update(occurrence=occurrence, frequency_bin=frequency, strata_status="complete")
            except ValueError as error:
                row["strata_reason"] = str(error)
            rows.append(row)
        return rows

    batches = [positions[i : i + batch_size] for i in range(0, len(positions), batch_size)]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(query, batches), 1):
            results.extend(rows)
            print(json.dumps(dict(point_batches_done=i, total=len(batches))), flush=True)
    merged = table.merge(
        pd.DataFrame(results), on=["lon", "lat"], how="left", validate="many_to_one"
    )
    merged.to_parquet(output / "station_strata.parquet", index=False)

    def input_dates():
        first = {
            sensor: collection(ee, sensor)
            .sort("system:time_start")
            .first()
            .get("system:time_start")
            .getInfo()
            for sensor in ("s1", "s2", "landsat")
        }
        return dict(
            first_utc={
                k: datetime.fromtimestamp(v / 1000, UTC).date().isoformat()
                for k, v in first.items()
            },
            sources={k: [s.collection for s in SOURCES[k]] for k in first},
        )

    dates = cached_json(output / "input_dates.json", input_dates)
    report = dict(
        complete=True,
        locations=len(merged),
        queried_positions=len(positions),
        source_location_bins=dict(Counter(merged.frequency_bin.fillna(-1).astype(int))),
        input_start=min(dates["first_utc"].values()),
        input_dates=dates,
        note="Source locations can repeat physical stations; no QC window counts yet",
    )
    atomic_json(output / "strata_report.json", report)
    return report
