"""Current checkpoint inference and bounded-memory, area-consistent regional mosaics."""

import json
import logging
import os
import shutil
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from affine import Affine
from numcodecs import Blosc
from rasterio.features import geometry_mask
from torch.utils.data import DataLoader, default_collate

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from hydrostate.data.regional.plan import epoch, load_plan
from hydrostate.data.retrieval.contract import PROFILE, TASKS, VERSION
from hydrostate.datasets.regional import RegionalDataset
from hydrostate.inference.regional_reports import check_fields, render_preview
from hydrostate.storage.packs import file_hash

SURFACE = dict(soil_moisture="land", et_water="water", et_ice="snow_ice", et_land="land")
LOG = logging.getLogger(__name__)


def load_model(config, checkpoint, pretrained, statistics, device):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    import hydrostate.models  # noqa: F401
    from hydrostate.cli import configure_data_contract
    from hydrostate.registry import MODELS

    cfg = Config.fromfile(config)
    configure_data_contract(cfg)
    if cfg.data_contract != VERSION or cfg.model.encoder.input_profile != PROFILE:
        raise ValueError("Regional inference requires the current retrieval input contract")
    if cfg.model.type != "MTLoRAHydroStateModel":
        raise ValueError("This regional exporter requires the current MTLoRA/shared-LoRA model")
    if not Path(pretrained).is_dir():
        raise FileNotFoundError(pretrained)
    stats = json.loads(Path(statistics).read_text())
    if stats.get("data_contract", {}).get("input_profile") != PROFILE:
        raise ValueError("Training statistics do not declare the current input profile")
    cfg.model.encoder.model_path = str(pretrained)
    cfg.model.encoder.gradient_checkpointing = False
    cfg.model.target_statistics = stats
    init_default_scope("hydrostate")
    model = MODELS.build(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().to(device)
    provenance = dict(
        checkpoint_sha256=file_hash(checkpoint),
        statistics_sha256=file_hash(statistics),
        model_config=cfg.model.to_dict(),
        input_profile=PROFILE,
        inference_code_sha256=file_hash(__file__),
    )
    if os.environ.get("HYDROSTATE_CODE_COMMIT"):
        provenance["code_commit"] = os.environ["HYDROSTATE_CODE_COMMIT"]
    return model, provenance


def predict_records(model, records, device):
    """One inference batch with bounded, grouped device-to-host transfers."""
    batch = default_collate(records)
    data = model.data_preprocessor(dict(inputs=batch["inputs"], data_samples={}), training=False)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), context:
        outputs = model.predict(data["inputs"])
    tasks = tuple(model.tasks)
    values = torch.stack([torch.stack([o[t] for t in tasks]) for o in outputs]).float().cpu()
    masks = ("prediction_valid", "soil_moisture_prediction_valid")
    validity = torch.stack([torch.stack([o[t] for t in masks]) for o in outputs]).bool().cpu()
    return [
        dict(zip(tasks, values[i]), **dict(zip(masks, validity[i]))) for i in range(len(outputs))
    ]


class RegionalMosaic:
    """Disk-backed sums; conditional surface rates use corresponding area weights."""

    def __init__(self, folder, shape, tasks, core):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.tasks, self.shape = tuple(tasks), tuple(shape)
        self.sums, self.weights = {}, {}
        for task in self.tasks:
            for name, arrays in (("sum", self.sums), ("weight", self.weights)):
                arrays[task] = np.memmap(
                    self.folder / f"{task}_{name}.bin", mode="w+", dtype="f4", shape=shape
                )
                arrays[task][:] = 0
        one = np.hanning(core + 2)[1:-1].astype("f4")
        self.kernel = np.outer(one, one)
        self.halo = (40 - core) // 2

    def add(self, row, col, output):
        h, core = self.halo, len(self.kernel)
        support = output["prediction_valid"][0].detach().cpu().numpy().astype(bool)
        values = {t: output[t][0].detach().float().cpu().numpy() for t in self.tasks}
        common = support[h : h + core, h : h + core]
        region = (
            slice(row // 3 + h, row // 3 + h + core),
            slice(col // 3 + h, col // 3 + h + core),
        )
        for task in self.tasks:
            value = values[task][h : h + core, h : h + core]
            valid = common.copy()
            if task == "soil_moisture":
                valid &= (
                    output["soil_moisture_prediction_valid"][0]
                    .detach()
                    .cpu()
                    .numpy()[h : h + core, h : h + core]
                )
            if not np.isfinite(value[valid]).all():
                raise ValueError(f"Nonfinite {task} on valid prediction support")
            weight = self.kernel * valid
            if task in SURFACE:
                surface = values[SURFACE[task]][h : h + core, h : h + core]
                weight = np.where(valid, weight * surface, 0)
            self.sums[task][region] += np.where(valid, value, 0) * weight
            self.weights[task][region] += weight

    def write(self, group, index, plan, min_land_fraction):
        grid, chunk = plan["grid"], 256
        summary = dict(inside_pixels=0, coverage_closure_max=0.0, variables={})
        tr = Affine(30, 0, grid["left"], 0, -30, grid["top"])
        for y in range(0, self.shape[0], chunk):
            for x in range(0, self.shape[1], chunk):
                region = (
                    slice(y, min(y + chunk, self.shape[0])),
                    slice(x, min(x + chunk, self.shape[1])),
                )
                shape = self.sums[self.tasks[0]][region].shape
                inside = geometry_mask(
                    [plan["geometry"]], shape, tr * Affine.translation(x, y), invert=True
                )
                values = {
                    t: np.divide(
                        self.sums[t][region],
                        self.weights[t][region],
                        out=np.full(shape, np.nan, "f4"),
                        where=self.weights[t][region] > 0,
                    )
                    for t in self.tasks
                }
                values["soil_moisture"][values["land"] <= min_land_fraction] = np.nan
                # Area weighting of each branch preserves the blended ET contributions.
                values["et_total"] = sum(
                    np.where(values[c] == 0, 0, values[c] * values[t])
                    for t, c in SURFACE.items()
                    if t.startswith("et_")
                )
                for value in values.values():
                    value[~inside] = np.nan
                check_fields(values, tolerance=0.01)
                coverage = sum(values[t] for t in ("water", "snow_ice", "land"))
                finite = np.isfinite(coverage)
                if finite.any():
                    summary["coverage_closure_max"] = max(
                        summary["coverage_closure_max"],
                        float(np.abs(coverage[finite] - 1).max()),
                    )
                summary["inside_pixels"] += int(inside.sum())
                for task, value in values.items():
                    selected = value[np.isfinite(value)]
                    stat = summary["variables"].setdefault(
                        task, dict(valid_pixels=0, sum=0.0, minimum=None, maximum=None)
                    )
                    if selected.size:
                        stat["valid_pixels"] += int(selected.size)
                        stat["sum"] += float(selected.sum(dtype="f8"))
                        stat["minimum"] = (
                            float(selected.min())
                            if stat["minimum"] is None
                            else min(stat["minimum"], float(selected.min()))
                        )
                        stat["maximum"] = (
                            float(selected.max())
                            if stat["maximum"] is None
                            else max(stat["maximum"], float(selected.max()))
                        )
                    group[task][(index, *region)] = value
                    group[task + "_valid"][(index, *region)] = np.isfinite(value).astype("u1")
        for arrays in (self.sums, self.weights):
            for value in arrays.values():
                value.flush()
        for stat in summary["variables"].values():
            total = stat.pop("sum")
            stat["mean"] = total / stat["valid_pixels"] if stat["valid_pixels"] else None
        return summary

    def close(self):
        self.sums.clear()
        self.weights.clear()
        shutil.rmtree(self.folder)


def predict(
    root,
    model,
    provenance,
    *,
    dates=None,
    batch_size=2,
    workers=0,
    device="cuda",
    pilot_dates=(),
    progress_seconds=60,
    display_surface_min=0.05,
):
    root, plan = Path(root), load_plan(root)
    if batch_size < 1 or workers < 0 or not set(TASKS) <= set(model.tasks):
        raise ValueError("Invalid batch settings or missing retrieval tasks")
    if progress_seconds <= 0 or not 0 <= display_surface_min <= 1:
        raise ValueError("Invalid progress interval or display surface threshold")
    if dates and not set(dates) <= {p["start"] for p in plan["periods"]}:
        raise ValueError("Requested dates must be target start dates in the regional plan")
    periods = [p for p in plan["periods"] if not dates or p["start"] in dates]
    if not set(pilot_dates) <= {p["start"] for p in periods}:
        raise ValueError("Pilot dates must belong to the selected target periods")
    priority = {date: i for i, date in enumerate(pilot_dates)}
    periods.sort(key=lambda p: priority.get(p["start"], len(priority)))
    for period in periods:
        if not (root / "manifests" / (period["start"] + ".json")).exists():
            raise FileNotFoundError(f"Inputs not ready for {period['start']}")
    identity = dict(plan=plan["identity"], provenance=provenance, mosaic="surface_area_cosine_v1")
    path = root / "products" / (fingerprint(identity)[:16] + ".zarr")
    path.parent.mkdir(exist_ok=True)
    with locked(str(path) + ".lock", blocking=False):
        group = zarr.open_group(str(path), mode="a")
        if "identity" in group.attrs and group.attrs["identity"] != identity:
            raise ValueError("Existing product belongs to a different model or input plan")
        shape = (len(plan["periods"]), plan["grid"]["height"] // 3, plan["grid"]["width"] // 3)
        tasks = tuple(model.tasks)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        group.attrs.update(
            identity=identity,
            crs=plan["grid"]["crs"],
            transform=[30, 0, plan["grid"]["left"], 0, -30, plan["grid"]["top"]],
            interpretation="Product-supervised retrieval; not independent 30 m truth",
        )
        for task in (*tasks, "et_total"):
            array = group.require_dataset(
                task,
                shape=shape,
                chunks=(1, 256, 256),
                dtype="f4",
                fill_value=np.nan,
                compressor=compressor,
            )
            array.attrs.update(
                _ARRAY_DIMENSIONS=["time", "y", "x"],
                units="mm/day"
                if task.startswith("et_")
                else "m3/m3"
                if task == "soil_moisture"
                else "1",
            )
            valid_array = group.require_dataset(
                task + "_valid",
                shape=shape,
                chunks=(1, 256, 256),
                dtype="u1",
                fill_value=0,
                compressor=compressor,
            )
            valid_array.attrs.update(_ARRAY_DIMENSIONS=["time", "y", "x"])
        complete = group.require_dataset(
            "date_complete", shape=(shape[0],), chunks=(1,), dtype="u1", fill_value=0
        )
        complete.attrs.update(_ARRAY_DIMENSIONS=["time"])
        times = np.array([[epoch(p["start"]), epoch(p["end"])] for p in plan["periods"]], "i8")
        if "time_bounds" not in group:
            group.create_dataset("time_bounds", data=times)
        group["time_bounds"].attrs.update(
            _ARRAY_DIMENSIONS=["time", "bounds"], units="seconds since 1970-01-01 00:00:00"
        )
        coords = dict(
            time=times.mean(axis=1).astype("i8"),
            x=plan["grid"]["left"] + (np.arange(shape[2]) + 0.5) * 30,
            y=plan["grid"]["top"] - (np.arange(shape[1]) + 0.5) * 30,
        )
        for name, values in coords.items():
            if name not in group:
                group.create_dataset(name, data=values, fill_value=None)
            group[name].attrs.update(
                _ARRAY_DIMENSIONS=[name],
                units="seconds since 1970-01-01 00:00:00" if name == "time" else "m",
            )
        group["time"].attrs["bounds"] = "time_bounds"
        progress_path = path.parent / f"{path.stem}_progress.json"
        run_started = time.monotonic()

        def progress(state, **details):
            atomic_json(
                progress_path,
                dict(
                    state=state,
                    product=str(path),
                    updated_utc=datetime.now(UTC).isoformat(),
                    completed_dates=int(complete[:].sum()),
                    planned_dates=shape[0],
                    selected_dates=len(periods),
                    batch_size=batch_size,
                    workers=workers,
                    elapsed_seconds=time.monotonic() - run_started,
                    **details,
                ),
            )

        progress("running")
        for period in periods:
            index = next(i for i, p in enumerate(plan["periods"]) if p["start"] == period["start"])
            if group["date_complete"][index]:
                if period["start"] in pilot_dates:
                    render_preview(path, index, display_surface_min=display_surface_min)
                continue
            dataset = RegionalDataset(root, period["start"], tasks)
            started = time.monotonic()
            reserve = json.loads((root / "drive.json").read_text())["min_free_gib"] * 1024**3
            # Disk-backed accumulators plus a conservative uncompressed output-date budget.
            required = int(np.prod(shape[1:])) * (len(tasks) * 8 + (len(tasks) + 1) * 5)
            if shutil.disk_usage(path.parent).free < reserve + required:
                raise OSError("Insufficient scratch space for this date's mosaic and output")
            loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=workers, collate_fn=list
            )
            mosaic = RegionalMosaic(
                path.parent / (path.stem + "_working"),
                shape[1:],
                tasks,
                plan["config"]["core_size"],
            )
            accepted = processed = 0
            last_report = started
            progress("reading", target=period["start"], windows=len(dataset))
            try:
                with torch.inference_mode():
                    for records in loader:
                        processed += len(records)
                        records = [r for r in records if r["accepted"]]
                        if records:
                            outputs = predict_records(model, records, device)
                            for record, output in zip(records, outputs):
                                row = dataset.rows[record["index"]]
                                mosaic.add(row["row"], row["col"], output)
                        accepted += len(records)
                        if time.monotonic() - last_report >= progress_seconds:
                            rate = processed / (time.monotonic() - started)
                            progress(
                                "predicting",
                                target=period["start"],
                                windows=len(dataset),
                                processed=processed,
                                accepted=accepted,
                                windows_per_second=rate,
                            )
                            LOG.info(
                                "%s: %d/%d windows, %d accepted, %.2f windows/s",
                                period["start"],
                                processed,
                                len(dataset),
                                accepted,
                                rate,
                            )
                            last_report = time.monotonic()
                coupling = model.water_soil_coupling
                progress("mosaicking", target=period["start"], accepted=accepted)
                audit = mosaic.write(
                    group, index, plan, coupling.min_land_fraction if coupling else 0.01
                )
                if not audit["variables"]["water"]["valid_pixels"]:
                    raise ValueError(f"No valid regional predictions for {period['start']}")
                atomic_json(
                    path.parent / f"{path.stem}_{period['start']}.json",
                    dict(
                        windows=len(dataset),
                        accepted=accepted,
                        target=period,
                        seconds=time.monotonic() - started,
                        audit=audit,
                    ),
                )
                group["date_complete"][index] = 1
                progress("date_complete", target=period["start"], accepted=accepted)
                if period["start"] in pilot_dates:
                    render_preview(path, index, display_surface_min=display_surface_min)
                LOG.info("Predicted %s: %d/%d windows", period["start"], accepted, len(dataset))
            except Exception as exc:
                progress("failed", target=period["start"], error=str(exc))
                raise
            finally:
                mosaic.close()
        zarr.consolidate_metadata(str(path))
        progress("complete" if complete[:].all() else "selected_dates_complete")
    return str(path)
