"""Small, explicitly unreleased station Zarr collection using production satellite inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.retrieval.alignment import grid_metadata, sensor_masks
from hydrostate.data.retrieval.catalog import cached_json
from hydrostate.data.retrieval.collector import input_arrays
from hydrostate.data.retrieval.contract import PROFILE
from hydrostate.data.spatial import grid
from hydrostate.storage import SampleShardWriter

from .contract import VERSION, SiteSchema


def site_grid(lon, lat):
    crs, _ = grid(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return dict(
        crs=crs,
        transform=[10, 0, x - 615, 0, -10, y + 615],
        block=[int(np.floor(lon)), int(np.floor(lat)), 1],
    )


def strata(ee, lon, lat, cache, climate_asset):
    def fetch():
        point = ee.Geometry.Point([lon, lat])
        occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        metadata = ee.Image("JRC/GSW1_4/Metadata").select(["valid_obs", "detections"])
        image = (
            occurrence.unmask(-1, False)
            .addBands(metadata.unmask(-1, False))
            .addBands(ee.Image(climate_asset).select(0).rename("climate_class"))
            .addBands(
                ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")
            )
        )
        return image.reduceRegion(ee.Reducer.first(), point, scale=30).getInfo()

    result = cached_json(
        cache / f"{fingerprint([lon, lat, climate_asset, 'metadata_v1'])}.json", fetch
    )
    occurrence = observed_occurrence(result)
    frequency = 0 if occurrence == 0 else 6 if occurrence == 100 else int(occurrence // 20) + 1
    return dict(
        result,
        occurrence=occurrence,
        frequency_bin=frequency,
        sampling_group=frequency,
        jrc_observed=True,
        strata_source="JRC_GSW1_4; ESA_WorldCover_v200; configured_Koppen_asset",
    )


def observed_occurrence(values):
    """Recover published masked never-water zeros only with positive valid-observation evidence."""
    if (values.get("valid_obs") or 0) <= 0:
        raise ValueError("Station centre has no JRC observations; it is not frequency zero")
    occurrence = values.get("occurrence")
    if occurrence is not None and 0 <= occurrence <= 100:
        return occurrence
    if values.get("detections") == 0:
        return 0
    raise ValueError("JRC occurrence missing despite water detections; centre requires review")


def collect_pilot(ee, records, root, config, norm, *, allow_empty=False):
    """Development-only samples; formal release still requires grouped splits and quotas."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(config, stage="pilot_unreleased", input_profile=PROFILE, version=VERSION)
    identity = fingerprint(
        {k: v for k, v in config.items() if k not in ("version", "input_profile")}
    )
    with locked(root / ".collection.lock", blocking=False):
        accepted = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
        path = root / "collection.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Pilot production configuration changed")
        atomic_json(path, config)
        for record in records:
            if not (record.get("sm") or {}).get("valid"):
                continue
            # Current pilot accepts known point SM; ET remains separately documented, never guessed.
            sample = dict(record, **site_grid(record["lon"], record["lat"]))
            sample.update(
                strata(
                    ee,
                    record["lon"],
                    record["lat"],
                    root / "cache" / "strata",
                    config["climate_asset"],
                )
            )
            sid = fingerprint(dict(record=record, config_sha256=identity))[:24]
            committed = root / "records" / f"{sid}.json"
            if committed.exists():
                continue
            rejection = root / "rejected" / f"{sid}.json"
            if rejection.exists():
                continue
            inputs, scenes, eligible = input_arrays(ee, sample, root / "cache", norm, config)
            masks = sensor_masks(inputs)
            observed = np.logical_or.reduce([v.any(axis=0)[0] for v in masks.values()])
            support = observed.reshape(40, 3, 40, 3).all(axis=(1, 3))
            if not eligible or not support[20, 20]:
                rejection = root / "rejected" / f"{sid}.json"
                rejection.parent.mkdir(exist_ok=True)
                atomic_json(rejection, dict(record=record, reason="input_or_point_coverage"))
                continue
            schema = SiteSchema()
            targets = {
                key: np.zeros(spec.tail_shape, spec.dtype) for key, spec in schema.targets.items()
            }
            targets["site_sm_native_value"][0] = record["sm"]["value"]
            targets["site_sm_native_valid"][0] = 1
            targets["site_sm_native_weights"][0, 20, 20] = 1
            base = root / "zarr" / sid
            writer = SampleShardWriter(
                str(base / "inputs.zarr"),
                str(base / "labels.zarr"),
                1,
                schema,
                attrs=dict(
                    config_sha256=identity,
                    input_profile=PROFILE,
                    input_grids=grid_metadata(sample),
                    scenes=scenes,
                    observation_records=[record],
                    normalization_hash=fingerprint(norm),
                ),
            )
            writer.write(inputs, targets)
            writer.finalize()
            row = dict(
                sample_id=sid,
                site_id=record["site_id"],
                group_id=record["group_id"],
                lon=record["lon"],
                lat=record["lat"],
                split=record.get("split", "train"),
                candidate_id=record.get("candidate_id", sid),
                version=VERSION,
                input_profile=PROFILE,
                crs=sample["crs"],
                transform=json.dumps(sample["transform"]),
                target_time=record["target_time"],
                target_end=record["target_end"],
                frequency_bin=sample["frequency_bin"],
                sampling_group=sample["frequency_bin"],
                climate_class=sample.get("climate_class", 0),
                landcover=sample.get("landcover", 0),
                observation_capacity=1,
                input_path=str((base / "inputs.zarr").relative_to(root)),
                label_path=str((base / "labels.zarr").relative_to(root)),
                shard_index=0,
            )
            committed.parent.mkdir(exist_ok=True)
            atomic_json(committed, row)
            accepted.append(row)
            print(json.dumps(dict(sample_id=sid, status="pilot_zarr_complete")), flush=True)
        if not accepted and not allow_empty:
            raise ValueError("No pilot samples passed input and station-label checks")
        if not accepted:
            return []
        temporary = root / "samples.parquet.part"
        pd.DataFrame(accepted).to_parquet(temporary, index=False)
        temporary.replace(root / "samples.parquet")
    return accepted
