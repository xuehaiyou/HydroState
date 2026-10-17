"""Checkpoint inference on fixed windows and resumable regional Zarr export."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import Blosc
from pyproj import CRS
from rasterio.features import rasterize
from torch.utils.data import DataLoader, Dataset

from hydrostate.data.gee.contracts import atomic_json, locked
from hydrostate.data.gee.selection import epoch
from hydrostate.data.gee.workflows.region import assemble_inputs
from hydrostate.inference.planning import aoi_mask, output_transform
from hydrostate.storage.packs import file_hash
from hydrostate.storage.scene_cache import SceneCache

LOG = logging.getLogger(__name__)
TASKS = ("water", "soil_moisture", "et")
FLAG_MEANINGS = (
    "prediction_supported soil_moisture_supported sentinel2_missing "
    "training_input_context_overlap training_target_period_overlap "
    "training_footprint_overlap open_water_et_unvalidated"
)


class WeightedMosaic:
    """Positive cosine weights over retained centres; no inpainting of invalid pixels."""

    def __init__(self, shape, core):
        self.sums = {k: np.zeros(shape, "f4") for k in TASKS}
        self.weights = {k: np.zeros(shape, "f4") for k in TASKS}
        self.squares = {k: np.zeros(shape, "f4") for k in TASKS}
        self.counts = {k: np.zeros(shape, "u2") for k in TASKS}
        one = np.hanning(core + 2)[1:-1].astype("f4")
        self.kernel = np.outer(one, one)

    def add(self, y, x, predictions):
        core = len(self.kernel)
        region = (slice(y, y + core), slice(x, x + core))
        for key, values in predictions.items():
            valid = np.isfinite(values)
            weight = self.kernel * valid
            value = np.where(valid, values, 0)
            self.sums[key][region] += weight * value
            self.squares[key][region] += weight * value**2
            self.weights[key][region] += weight
            self.counts[key][region] += valid

    def finish(self, min_land_fraction=0.01):
        result = {
            k: np.divide(
                self.sums[k],
                self.weights[k],
                out=np.full_like(self.sums[k], np.nan),
                where=self.weights[k] > 0,
            )
            for k in TASKS
        }
        # Apply the original WF-SM domain after blending as well as within each patch.
        valid_land = np.isfinite(result["water"]) & ((1 - result["water"]) > min_land_fraction)
        result["soil_moisture"][~valid_land] = np.nan
        return result

    def disagreement(self):
        """Window-context spread is a diagnostic, not calibrated predictive uncertainty."""
        summary = {}
        for key in TASKS:
            valid = (self.counts[key] > 1) & (self.weights[key] > 0)
            weights = self.weights[key][valid]
            variance = self.squares[key][valid] / weights - (self.sums[key][valid] / weights) ** 2
            spread = np.sqrt(np.maximum(variance, 0))
            summary[key] = dict(
                count=int(valid.sum()),
                mean_std=float(spread.mean()) if len(spread) else None,
                p95_std=float(np.quantile(spread, 0.95)) if len(spread) else None,
            )
        return summary


class RegionDataset(Dataset):
    def __init__(self, root, plan, table, drop_s2=False):
        self.root, self.plan = root, plan
        self.rows = table.to_dict("records")
        self.drop_s2, self.cache = drop_s2, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.cache is None:
            self.cache = SceneCache(self.root, self.plan)
        arrays = assemble_inputs(self.cache, self.rows[index], drop_s2=self.drop_s2)
        return dict(inputs={k: torch.from_numpy(v) for k, v in arrays.items()}, index=index)


def masked_outputs(model, inputs):
    """Run the original model with its trained preprocessing and predicted WF mask."""
    batch = inputs["s1"].shape[0]
    data = model.data_preprocessor(
        dict(
            inputs=inputs,
            data_samples=dict(
                water=torch.zeros(batch, 1, 120, 120),
                water_valid=torch.zeros(batch, 1, 120, 120, dtype=torch.bool),
            ),
        ),
        training=False,
    )
    outputs = model(data["inputs"], mode="tensor")
    support = data["data_samples"]["prediction_valid"]
    if any(not torch.isfinite(value)[support].all() for value in outputs.values()):
        raise ValueError("Nonfinite model output on observed support")
    sm = model.water_soil_coupling.prepare(outputs, dict(prediction_valid=support))
    result = {
        k: v.float().masked_fill(
            ~(sm.pixel_valid if k == "soil_moisture" else support), float("nan")
        )
        for k, v in outputs.items()
    }
    return {k: v[:, 0].cpu().numpy() for k, v in result.items()}


def load_model(plan, *, checkpoint=None, pretrained=None, statistics=None, device="cuda"):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings

    import hydrostate.models  # noqa: F401
    from hydrostate.registry import MODELS

    init_default_scope("hydrostate")

    config = plan["config"]
    checkpoint = str(checkpoint or config["checkpoint"])
    pretrained = str(pretrained or config["pretrained"])
    statistics = str(statistics or config["target_statistics"])
    if not Path(pretrained).is_dir():
        raise FileNotFoundError(f"Local pretrained model directory is required: {pretrained}")
    cfg = Config.fromfile(config["model_config"])
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.encoder.model_path = pretrained
    cfg.model.encoder.gradient_checkpointing = False
    cfg.model.target_statistics = json.loads(Path(statistics).read_text())
    if cfg.model.type != "MultiLevelHydroStateModel":
        raise ValueError(
            "This regional contract is for the original three-variable multilevel model"
        )
    model = MODELS.build(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    if tuple(model.tasks) != TASKS or model.output_scale != 3:
        raise ValueError("Checkpoint task/grid contract mismatch")
    model.eval().to(device)
    provenance = dict(
        checkpoint=checkpoint,
        checkpoint_sha256=file_hash(checkpoint),
        statistics_sha256=file_hash(statistics),
        model_config=cfg.model.to_dict(),
        torch_version=torch.__version__,
        device=device,
    )
    import os

    provenance["code_commit"] = os.environ.get("HYDROSTATE_CODE_COMMIT", "local-development")
    provenance["inference_contract"] = "fixed120_center32_cosine_v1"
    return model, provenance


def create_product(root, plan, provenance, *, drop_s2=False):
    folder = Path(root) / "products"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("poyang_3vars_without_s2.zarr" if drop_s2 else "poyang_3vars.zarr")
    identity = dict(
        plan_hash=plan["plan_hash"],
        checkpoint_sha256=provenance["checkpoint_sha256"],
        statistics_sha256=provenance["statistics_sha256"],
        drop_s2=drop_s2,
    )
    if (path / ".zattrs").exists():
        group = zarr.open_group(str(path), mode="r+")
        if group.attrs.get("identity") != identity:
            raise ValueError("Output belongs to a different checkpoint/plan")
        return group
    group = zarr.open_group(str(path), mode="a")
    grid, n = plan["grid"], len(plan["periods"])
    h, w, core = grid["height"], grid["width"], grid["core"]
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    def create(name, shape, chunks, dtype, dimensions, *, fill_value=None, **attrs):
        array = group.require_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
        )
        array.attrs.update(_ARRAY_DIMENSIONS=dimensions, **attrs)
        return array

    t = create(
        "time",
        (n,),
        (n,),
        "i8",
        ["time"],
        units="seconds since 1970-01-01 00:00:00",
        calendar="proleptic_gregorian",
        bounds="time_bounds",
        standard_name="time",
    )
    t[:] = [epoch(p["time"]) for p in plan["periods"]]
    bounds = create("time_bounds", (n, 2), (n, 2), "i8", ["time", "bounds"], units=t.attrs["units"])
    bounds[:] = [[epoch(p["start"]), epoch(p["end"])] for p in plan["periods"]]
    create(
        "x", (w,), (min(w, 4096),), "f8", ["x"], units="m", standard_name="projection_x_coordinate"
    )[:] = grid["left"] + (np.arange(w) + 0.5) * 30
    create(
        "y", (h,), (min(h, 4096),), "f8", ["y"], units="m", standard_name="projection_y_coordinate"
    )[:] = grid["top"] - (np.arange(h) + 0.5) * 30
    crs = CRS.from_user_input(grid["crs"])
    create("spatial_ref", (), (), "i4", [], spatial_ref=crs.to_wkt(), **crs.to_cf())[...] = 0
    group["spatial_ref"].attrs["GeoTransform"] = " ".join(
        map(str, output_transform(grid).to_gdal())
    )
    for task, units in zip(TASKS, ("1", "m3 m-3", "mm day-1")):
        create(
            task,
            (n, h, w),
            (1, 512, 512),
            "f4",
            ["time", "y", "x"],
            fill_value=np.nan,
            units=units,
            grid_mapping="spatial_ref",
            temporal_support=(
                "8-day mean daily ET"
                if task == "et"
                else "reference-date estimate from preceding 63 days"
            ),
        )
    create(
        "quality",
        (n, h, w),
        (1, 512, 512),
        "u1",
        ["time", "y", "x"],
        fill_value=0,
        flag_masks=[1, 2, 4, 8, 16, 32, 64],
        flag_meanings=FLAG_MEANINGS,
    )
    create("aoi", (h, w), (512, 512), "u1", ["y", "x"])[:] = aoi_mask(plan)
    records = [(r["geometry"], 1) for r in plan["training"]]
    overlap = (
        rasterize(records, (h, w), transform=output_transform(grid), dtype="uint8")
        if records
        else np.zeros((h, w), "u1")
    )
    create("training_footprint_overlap", (h, w), (512, 512), "u1", ["y", "x"])[:] = overlap
    stride = grid.get("stride", core)
    ny, nx = (h - core) // stride + 1, (w - core) // stride + 1
    create(
        "observation_time",
        (n, ny, nx, 4, 3),
        (1, ny, nx, 4, 3),
        "i8",
        ["time", "window_y", "window_x", "slot", "sensor"],
        fill_value=0,
        units="seconds since 1970-01-01 00:00:00",
        sensor_order=list(TASK_SENSOR_KEYS),
        missing_value=0,
    )
    create(
        "window_accepted",
        (n, ny, nx),
        (1, ny, nx),
        "u1",
        ["time", "window_y", "window_x"],
        fill_value=0,
    )
    create("date_complete", (n,), (1,), "u1", ["time"], fill_value=0)
    group.attrs.update(
        identity=identity,
        provenance=provenance,
        Conventions="CF-1.8",
        plan=plan,
        storage="ordinary directory Zarr v2",
        warning="30 m output is not independently validated 30 m SM/ET resolution",
    )
    return group


TASK_SENSOR_KEYS = ("s1", "s2", "landsat")


def write_date(group, plan, index, maps, table):
    grid = plan["grid"]
    aoi = group["aoi"][:].astype(bool)
    quality = np.zeros(aoi.shape, dtype="u1")
    supported = np.isfinite(maps["water"]) & np.isfinite(maps["et"]) & aoi
    quality[supported] |= 1
    quality[np.isfinite(maps["soil_moisture"]) & aoi] |= 2
    quality[(group["training_footprint_overlap"][:] > 0) & aoi] |= 32
    quality[(maps["water"] >= 0.99) & supported] |= 64
    core = grid["core"]
    stride = grid.get("stride", core)
    observation = np.zeros(group["observation_time"].shape[1:], dtype="i8")
    accepted = np.zeros(group["window_accepted"].shape[1:], dtype="u1")
    drop_s2 = group.attrs["identity"]["drop_s2"]
    for record in table.to_dict("records"):
        row, col = record["row"], record["col"]
        selection = json.loads(record["selections"])
        wy, wx = row // stride, col // stride
        accepted[wy, wx] = record["accepted"]
        for slot in range(4):
            for mi, key in enumerate(TASK_SENSOR_KEYS):
                scene = selection[f"{key}_{slot}"]
                if scene and not (drop_s2 and key == "s2"):
                    observation[wy, wx, slot, mi] = scene["time"]
        flags = 4 if not observation[wy, wx, :, 1].any() else 0
        flags |= 8 if json.loads(record["training_context_ids"]) else 0
        flags |= 16 if json.loads(record["training_target_ids"]) else 0
        quality[row : row + core, col : col + core] |= flags
    quality[~aoi] = 0
    for key, values in maps.items():
        group[key][index] = np.where(aoi, values, np.nan)
    group["quality"][index] = quality
    group["observation_time"][index] = observation
    group["window_accepted"][index] = accepted
    # Final commit marker: readers must ignore incomplete dates.
    group["date_complete"][index] = 1
    zarr.consolidate_metadata(group.store)
    return dict(
        time=plan["periods"][index]["time"],
        windows=len(table),
        accepted_windows=int(table.accepted.sum()),
        aoi_pixels=int(aoi.sum()),
        valid_pixels={k: int((np.isfinite(v) & aoi).sum()) for k, v in maps.items()},
        ranges={
            k: [float(np.nanmin(v)), float(np.nanmax(v))] if np.isfinite(v).any() else None
            for k, v in maps.items()
        },
    )


def predict_dates(
    root,
    plan,
    dates=None,
    *,
    batch_size=8,
    workers=2,
    device="cuda",
    checkpoint=None,
    pretrained=None,
    statistics=None,
    drop_s2=False,
):
    root = Path(root)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA allocation is required")
    requested = set(dates or [p["time"] for p in plan["periods"]])
    with locked(root / "predict.lock", blocking=False):
        model, provenance = load_model(
            plan, checkpoint=checkpoint, pretrained=pretrained, statistics=statistics, device=device
        )
        group = create_product(root, plan, provenance, drop_s2=drop_s2)
        grid, halo, core = plan["grid"], plan["grid"]["halo"], plan["grid"]["core"]
        for index, period in enumerate(plan["periods"]):
            if period["time"] not in requested or group["date_complete"][index]:
                continue
            started = time.perf_counter()
            table = pd.read_parquet(root / "manifests" / f"{period['time']}.parquet")
            accepted = table.loc[table.accepted].reset_index(drop=True)
            if drop_s2:
                # An ablation cannot run if S2 was the only available modality.
                accepted = accepted.loc[
                    accepted.selections.map(
                        lambda s: any(
                            v for k, v in json.loads(s).items() if not k.startswith("s2_")
                        )
                    )
                ].reset_index(drop=True)
            dataset = RegionDataset(root, plan, accepted, drop_s2=drop_s2)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=device.startswith("cuda"),
            )
            mosaic = WeightedMosaic((grid["height"], grid["width"]), core)
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                    ):
                        output = masked_outputs(model, batch["inputs"])
                    for j, item_index in enumerate(batch["index"].tolist()):
                        row = dataset.rows[item_index]
                        y, x = row["row"], row["col"]
                        mosaic.add(
                            y,
                            x,
                            {
                                key: output[key][j, halo : halo + core, halo : halo + core]
                                for key in TASKS
                            },
                        )
                    LOG.info("%s batch %d/%d", period["time"], batch_index + 1, len(loader))
            maps = mosaic.finish(model.water_soil_coupling.min_land_fraction)
            report = write_date(group, plan, index, maps, table)
            report["window_context_disagreement"] = mosaic.disagreement()
            report.update(
                elapsed_seconds=time.perf_counter() - started,
                completed_at=datetime.now(UTC).isoformat(),
                drop_s2=drop_s2,
            )
            destination = root / "reports"
            destination.mkdir(exist_ok=True)
            suffix = "_without_s2" if drop_s2 else ""
            atomic_json(destination / f"{period['time']}{suffix}.json", report)
            LOG.info("Completed %s", json.dumps(report))
