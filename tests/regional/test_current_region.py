import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from affine import Affine

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.regional.pipeline import capacity, publish_ready, status
from hydrostate.data.regional.plan import block_grid, blocks_for_window, initialize, product
from hydrostate.data.retrieval.composites import normalize, radar_orbit
from hydrostate.data.retrieval.contract import INPUT_GROUPS, PROFILE, TASKS, VERSION
from hydrostate.datasets.regional import RegionalDataset
from hydrostate.inference.regional import RegionalMosaic, load_model, predict
from hydrostate.storage.regional import commit, ingest, product_path, ready


def config():
    return json.loads(Path("configs/inference/poyang_pilot.json").read_text())


@pytest.mark.parametrize("extent", [0, 0.03])
def test_two_pixel_overlap_covers_aoi_and_aligns_native_grids(tmp_path, extent):
    from rasterio.features import geometry_mask

    cfg = config()
    cfg.update(core_size=36, stride=34)
    cfg["aoi"]["bbox"][2] += extent
    cfg["aoi"]["bbox"][3] += extent
    plan = initialize(tmp_path, cfg)
    grid = plan["grid"]
    shape = (grid["height"] // 3, grid["width"] // 3)
    counts = np.zeros(shape, dtype="u1")
    for row in pd.read_parquet(tmp_path / "windows.parquet").to_dict("records"):
        # Even 10 m offsets align with every native 10/15/20/30/60 m grid.
        assert row["row"] % 6 == row["col"] % 6 == 0
        y, x = row["row"] // 3 + 2, row["col"] // 3 + 2
        assert y + 36 <= shape[0] and x + 36 <= shape[1]
        counts[y : y + 36, x : x + 36] += 1
    aoi = geometry_mask(
        [plan["geometry"]], shape,
        Affine(30, 0, grid["left"], 0, -30, grid["top"]), invert=True,
    )
    assert aoi.any()
    assert (counts[aoi] > 0).all()
    assert counts.max() == (4 if extent else 1)


def populate(root, *, cross_blocks=False, empty=False):
    cfg = config()
    if cross_blocks:
        cfg["aoi"]["bbox"][2] += 0.012
    plan = initialize(root, cfg)
    windows = pd.read_parquet(root / "windows.parquet")
    windows["orbit"] = json.dumps([25, "ASCENDING"])
    period = plan["periods"][0]
    products = {}
    for row in windows.to_dict("records"):
        for block in blocks_for_window(plan, row["row"], row["col"]):
            for name, spec in INPUT_GROUPS.items():
                for slot in period["slots"]:
                    item = product(
                        plan, name, slot, block, [25, "ASCENDING"] if spec.sensor == "s1" else None
                    )
                    if item["id"] in products:
                        continue
                    grid = block_grid(plan, block, spec.resolution)
                    raw = np.zeros((len(spec.bands) + 4, grid["height"], grid["width"]), "f4")
                    if not empty:
                        raw[-4:] = np.array([2, 3, 1, 5])[:, None, None]
                    commit(root, plan, item, raw)
                    products[item["id"]] = item
    target = root / "targets"
    target.mkdir()
    windows.to_parquet(target / (period["start"] + ".parquet"), index=False)
    atomic_json(
        target / (period["start"] + ".json"),
        dict(plan=plan["identity"], period=period, products=list(products.values())),
    )
    publish_ready(root, plan)
    return plan, list(products.values())


def test_native_cross_block_reader_and_missing_sensor_support(tmp_path):
    plan, products = populate(tmp_path, cross_blocks=True)
    ds = RegionalDataset(tmp_path, plan["periods"][0]["start"])
    index = next(i for i, r in enumerate(ds.rows) if r["col"] % 120 != 0)
    raw, accepted = ds.native_inputs(index)
    assert accepted
    for name, spec in INPUT_GROUPS.items():
        expected = normalize(
            spec.sensor,
            np.zeros((len(spec.bands), *spec.shape()), "f4"),
            np.ones(spec.shape(), bool),
            plan["normalization"],
            spec.bands,
        )
        np.testing.assert_array_equal(raw[name][0], expected)
        assert raw[name + "_valid"].all()
        assert (raw[name + "_quality"][:, 0] == 2).all()
    sample = ds[index]
    assert sample["inputs"]["s2"].shape == (5, 12, 120, 120)
    assert set(sample["inputs"]["target_time"]) == set(TASKS)
    assert not sample["data_samples"]
    uncached = RegionalDataset(tmp_path, plan["periods"][0]["start"], cache_mib=0)
    for i in ds.spatial_indices():
        original, expected_accepted = uncached.native_inputs(i)
        cached, actual_accepted = ds.native_inputs(i)
        assert actual_accepted == expected_accepted
        for key in original:
            np.testing.assert_array_equal(cached[key], original[key])
    assert sorted(ds.spatial_indices()) == list(range(len(ds)))
    assert status(tmp_path, audit=True)["ready_dates"] == 1
    # A transport failure must never become valid empty observations.
    (product_path(tmp_path, products[0]) / "_SUCCESS.json").unlink()
    with pytest.raises(ValueError, match="Missing product"):
        status(tmp_path, audit=True)


def test_offline_retiling_reads_old_arrays_without_changing_source(monkeypatch, tmp_path):
    from shapely.geometry import box, mapping

    from hydrostate.data.regional import retiling

    source, target = tmp_path / "source", tmp_path / "retiled"
    plan, _ = populate(source, cross_blocks=True)
    frozen = (source / "plan.json").read_bytes()
    day = plan["periods"][0]["start"]
    monkeypatch.setattr(retiling, "catalog", lambda *a: [dict(
        geometry=mapping(box(-180, -90, 180, 90)),
        properties=dict(relativeOrbitNumber_start=25, orbitProperties_pass="ASCENDING"),
    )])
    result = retiling.retile(target, source, [day])
    assert result["windows"] > 1
    assert (source / "plan.json").read_bytes() == frozen
    ds = RegionalDataset(target, day)
    baseline, _ = RegionalDataset(source, day).native_inputs(0)
    for index in range(len(ds)):
        raw, accepted = ds.native_inputs(index)
        assert accepted
        for key in raw:
            np.testing.assert_array_equal(raw[key], baseline[key])
    assert not (target / "inputs").exists()
    assert ds.input_root == source.resolve()
    for item in json.loads((target / "manifests" / f"{day}.json").read_text())["products"]:
        assert (product_path(source, item) / "_SUCCESS.json").exists()
    assert status(target, audit=True)["ready_dates"] == 1
    assert retiling.retile(target, source, [day]) == result
    from hydrostate.data.regional import pipeline, pixels

    for prepare in (pipeline.prepare, pixels.prepare):
        with pytest.raises(ValueError, match="read-only"):
            prepare(target)
    # A ready view cannot silently read a mutated source plan.
    from hydrostate.data.gee.contracts import fingerprint

    changed = json.loads(frozen)
    changed["grid"]["left"] += 60
    changed.pop("identity")
    changed["identity"] = fingerprint(changed)
    atomic_json(source / "plan.json", changed)
    with pytest.raises(ValueError, match="Referenced input plan"):
        RegionalDataset(target, day)


def test_empty_observations_are_complete_but_not_predictable(tmp_path):
    plan, _ = populate(tmp_path, empty=True)
    ds = RegionalDataset(tmp_path, plan["periods"][0]["start"])
    assert not ds[0]["accepted"]
    assert not ds[0]["inputs"]["sensor_valid"].any()
    assert status(tmp_path, audit=True)["ready_dates"] == 1


def test_year_boundary_and_orbit_identity(tmp_path):
    cfg = config()
    cfg.update(start="2020-12-26", end="2021-01-02")
    plan = initialize(tmp_path, cfg)
    assert [p["start"] for p in plan["periods"]] == ["2020-12-26", "2021-01-01"]
    assert plan["periods"][0]["end"] == "2021-01-01"
    assert plan["periods"][0]["slots"][2] == ["2020-12-26", "2021-01-03"]
    props = [
        dict(relativeOrbitNumber_start=25, orbitProperties_pass="ASCENDING"),
        dict(relativeOrbitNumber_start=25, orbitProperties_pass="DESCENDING"),
    ]
    assert radar_orbit(props) == radar_orbit(props[::-1])
    p = plan["periods"][0]
    assert (
        product(plan, "s1_10m", p["slots"][0], [0, 0], [25, "ASCENDING"])["id"]
        != product(plan, "s1_10m", p["slots"][0], [0, 0], [25, "DESCENDING"])["id"]
    )


def test_geotiff_ingestion_rejects_shift_and_preserves_zero(tmp_path):
    import rasterio

    plan = initialize(tmp_path, config())
    item = product(plan, "s1_10m", plan["periods"][0]["slots"][0], [0, 0], [25, "ASCENDING"])
    grid = block_grid(plan, [0, 0])
    bands = ["t0_" + b for b in ("VV", "VH", "days", "mean_day", "first_day", "last_day")]
    raw = np.zeros((6, grid["height"], grid["width"]), "f4")
    raw[2:] = np.array([1, 0, 0, 0])[:, None, None]
    path = tmp_path / "part.tif"
    for shift in (0.5, 0):
        tr = Affine(*grid["transform"]) * Affine.translation(shift, 0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=6,
            width=grid["width"],
            height=grid["height"],
            dtype="float32",
            crs=grid["crs"],
            transform=tr,
        ) as f:
            f.write(raw)
            f.descriptions = tuple(bands)
        job = dict(grid=grid, products=[item], bands=bands)
        if shift:
            with pytest.raises(ValueError, match="not pixel aligned"):
                ingest(tmp_path, plan, job, [path])
            assert not ready(tmp_path, item)
        else:
            ingest(tmp_path, plan, job, [path])
            assert ready(tmp_path, item)
            assert zarr.open_group(str(product_path(tmp_path, item)), mode="r")["valid"][:].all()


def test_capacity_reports_specific_resource():
    settings = config()["drive"]
    gib = 1024**3
    assert capacity(settings, gib / 2, 0, 0, 100 * gib)["blocked_by"] == ["google_drive"]
    assert capacity(settings, gib / 2, 0, 100 * gib, 0)["blocked_by"] == ["hpc_scratch"]
    assert capacity(settings, 2 * gib, 0, 100 * gib, 100 * gib)["blocked_by"] == [
        "pending_export_budget"
    ]


@pytest.mark.parametrize("private_rank", [0, 1])
def test_current_model_checkpoint_to_regional_product(monkeypatch, tmp_path, private_rank):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig
    from rslearn.models.olmoearth_pretrain import model as wrapper

    import hydrostate.models  # noqa: F401
    from hydrostate.cli import configure_data_contract
    from hydrostate.normalization import CONFIGURED_SUPPORT_VERSION
    from hydrostate.registry import MODELS
    from hydrostate.tasks import task_config

    monkeypatch.setattr(
        wrapper,
        "load_model_from_path",
        lambda *a, **kw: SimpleNamespace(
            encoder=EncoderConfig(
                supported_modality_names=["sentinel1", "sentinel2_l2a", "landsat"],
                embedding_size=16,
                num_heads=2,
                depth=3,
                drop_path=0,
                num_register_tokens=2,
                use_flash_attn=False,
                use_linear_patch_embed=True,
                position_encoding="rope_3d_mixed",
            ).build()
        ),
    )
    input_root = tmp_path / ("source" if private_rank == 0 else "inputs")
    plan, _ = populate(input_root)
    if private_rank == 0:
        from shapely.geometry import box, mapping

        from hydrostate.data.regional import retiling

        monkeypatch.setattr(retiling, "catalog", lambda *a: [dict(
            geometry=mapping(box(-180, -90, 180, 90)),
            properties=dict(relativeOrbitNumber_start=25, orbitProperties_pass="ASCENDING"),
        )])
        retiling.retile(tmp_path / "inputs", input_root, [plan["periods"][0]["start"]])
    cfg = Config.fromfile("configs/hydrostate_mtlora.py")
    configure_data_contract(cfg)
    stats = dict(
        version=2,
        split="train",
        ddof=0,
        support_version=CONFIGURED_SUPPORT_VERSION,
        training_fingerprint="0" * 64,
        task_specs=task_config(TASKS),
        data_contract=dict(version=VERSION, input_profile=PROFILE),
        tasks={t: dict(count=100, mean=0.3, std=0.2) for t in TASKS},
    )
    cfg.model.encoder.update(
        model_path=str(tmp_path),
        embedding_channels=16,
        out_indices=[0, 1, 2],
        shared_rank=2,
        private_rank=private_rank,
        gradient_checkpointing=False,
    )
    cfg.model.decoder.update(in_channels=16, channels=8)
    cfg.model.input_fusion.feature_channels = 16
    for head in cfg.model.heads.values():
        head.in_channels = 8
    cfg.model.target_statistics = stats
    init_default_scope("hydrostate")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = MODELS.build(cfg.model).eval()
        checkpoint = tmp_path / "checkpoint.pth"
        torch.save(dict(state_dict=model.state_dict()), checkpoint)
        cfg.dump(str(tmp_path / "model.py"))
        atomic_json(tmp_path / "stats.json", stats)
        model, provenance = load_model(
            tmp_path / "model.py", checkpoint, tmp_path, tmp_path / "stats.json", "cpu"
        )
        date = plan["periods"][0]["start"]
        path = predict(
            tmp_path / "inputs", model, provenance, device="cpu", batch_size=1, pilot_dates=[date]
        )
        group = zarr.open_group(path, mode="r")
        assert group["date_complete"][:].tolist() == [1]
        valid = group["et_total_valid"][0].astype(bool)
        assert valid.any()
        np.testing.assert_allclose(
            sum(group[k][0][valid] for k in ("water", "snow_ice", "land")), 1, atol=1e-6
        )
        np.testing.assert_allclose(
            group["et_total"][0][valid],
            sum(
                group[c][0][valid] * group[t][0][valid]
                for t, c in (("et_water", "water"), ("et_ice", "snow_ice"), ("et_land", "land"))
            ),
            rtol=1e-6,
        )
        product = Path(path)
        assert (product.parent / f"{product.stem}_previews" / f"{date}.png").is_file()
        audit_path = product.parent / f"{product.stem}_{date}.json"
        audit = json.loads(audit_path.read_text())
        assert audit["audit"]["coverage_closure_max"] < 1e-6
        assert audit["audit"]["variables"]["et_total"]["valid_pixels"] == int(valid.sum())
        mtime = audit_path.stat().st_mtime_ns
        assert predict(tmp_path / "inputs", model, provenance, device="cpu") == path
        assert audit_path.stat().st_mtime_ns == mtime
        progress = json.loads((product.parent / f"{product.stem}_progress.json").read_text())
        assert progress["state"] == "complete"
        assert progress["completed_dates"] == 1
    finally:
        torch.set_num_threads(threads)


def test_surface_weighted_mosaic_preserves_total_and_conditional_sm(tmp_path):
    plan = initialize(tmp_path / "region", config())
    mosaic = RegionalMosaic(tmp_path / "work", (40, 40), TASKS, 32)
    for water, land, ew, el, sm in ((0.5, 0.5, 10.0, 2.0, 0.2), (0.25, 0.75, 2.0, 6.0, 0.4)):
        values = dict(
            water=water,
            snow_ice=0.0,
            land=land,
            soil_moisture=sm,
            et_water=ew,
            et_ice=1.0,
            et_land=el,
        )
        output = {k: torch.full((1, 40, 40), v) for k, v in values.items()}
        output["prediction_valid"] = torch.ones((1, 40, 40), dtype=torch.bool)
        output["soil_moisture_prediction_valid"] = output["prediction_valid"]
        mosaic.add(0, 0, output)
    group = zarr.open_group(str(tmp_path / "product"), mode="w")
    for key in (*TASKS, "et_total"):
        group.create_dataset(key, shape=(1, 40, 40), dtype="f4", fill_value=np.nan)
        group.create_dataset(key + "_valid", shape=(1, 40, 40), dtype="u1")
    mosaic.write(group, 0, plan, 0.01)
    valid = group["et_total_valid"][0].astype(bool)
    assert valid.any()
    np.testing.assert_allclose(group["et_total"][0][valid], 5.5, rtol=1e-6)
    np.testing.assert_allclose(group["soil_moisture"][0][valid], 0.32, rtol=1e-6)
    assert not group["et_ice_valid"][:].any()
    mosaic.close()


def test_adjacent_targets_share_slots_and_keep_s1_orbit(monkeypatch, tmp_path):
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform

    from hydrostate.data.regional import exports
    from hydrostate.data.regional.plan import epoch

    cfg = config()
    cfg["end"] = "2020-07-20"
    plan = initialize(tmp_path, cfg)
    geometry = mapping(
        transform(
            Transformer.from_crs(plan["grid"]["crs"], 4326, always_xy=True).transform,
            shape(plan["geometry"]).buffer(2000),
        )
    )

    def catalog(ee, root, plan, sensor, start, end):
        return [
            dict(
                geometry=geometry,
                properties=dict(
                    asset_id=sensor + "_fake",
                    system_time=0,
                    **{"system:time_start": epoch(start) * 1000},
                    relativeOrbitNumber_start=25,
                    orbitProperties_pass="ASCENDING",
                ),
            )
        ]

    monkeypatch.setattr(exports, "catalog", catalog)
    planned = list(
        exports.jobs(None, tmp_path, plan, max_bytes=cfg["drive"]["max_export_gib"] * 1024**3)
    )
    products = [p for j in planned for p in j["products"]]
    assert len(products) == 6 * len(INPUT_GROUPS)
    assert len({p["id"] for p in products}) == len(products)
    assert all(p["orbit"] == [25, "ASCENDING"] for p in products if p["group"] == "s1_10m")
    budget = 700_000
    split = list(exports.jobs(None, tmp_path, plan, max_bytes=budget))
    assert all(j["raw_bytes"] <= budget for j in split)
    assert {p["id"] for j in split for p in j["products"]} == {p["id"] for p in products}
    with pytest.raises(ValueError, match="One native slot exceeds"):
        list(exports.jobs(None, tmp_path, plan, max_bytes=1))


@pytest.mark.parametrize("drain", [False, True])
@pytest.mark.parametrize("unknown_id", [False, True])
def test_queue_recovers_submitted_task_without_duplicate(monkeypatch, tmp_path, drain, unknown_id):
    from hydrostate.data.gee.state import ExportState
    from hydrostate.data.regional import pipeline

    region_config = config()
    if drain:
        region_config["end"] = "2020-07-20"
    plan = initialize(tmp_path, region_config)
    cfg = config()["drive"]
    cfg["poll_seconds"] = 0.001
    atomic_json(tmp_path / "drive.json", cfg)
    item = product(plan, "s1_10m", plan["periods"][0]["slots"][0], [0, 0], [25, "ASCENDING"])
    job = dict(id="existing", products=[item], raw_bytes=100)
    (tmp_path / "targets").mkdir()
    atomic_json(
        tmp_path / "targets" / (plan["periods"][0]["start"] + ".json"),
        dict(plan=plan["identity"], period=plan["periods"][0], products=[item]),
    )
    state = ExportState(tmp_path / "exports.sqlite")
    state.add(job)
    state.update("existing", status="submitted", task_id="original_task")
    state.close()
    calls = []

    class Transport:
        def __init__(self, *a, **kw):
            pass

        def ensure_folder(self):
            pass

        def status(self, task):
            if unknown_id and task == "original_task":
                return dict(state="UNKNOWN")
            assert task == ("resolved_task" if unknown_id else "original_task")
            return dict(state="COMPLETED")

        def reconcile(self, job):
            assert unknown_id and job == "existing"
            return "resolved_task"

        def inventory(self, job):
            return [dict(id="file", size=100)]

        def submit(self, *a):
            raise AssertionError("duplicate submission")

        def cleanup(self, record):
            assert ready(tmp_path, item)
            calls.append(record["id"])

    def receive(*args):
        raw = np.zeros((6, 120, 120), "f4")
        commit(tmp_path, plan, item, raw)
        return dict(bytes=100)

    monkeypatch.setattr(pipeline, "initialize", lambda *a: None)
    monkeypatch.setattr(pipeline, "CompositeDriveTransport", Transport)

    def jobs(*a, **kw):
        assert not drain, "Draining must not plan new exports"
        return iter(())

    monkeypatch.setattr(pipeline, "jobs", jobs)
    monkeypatch.setattr(pipeline, "receive", receive)
    report = pipeline.prepare(tmp_path, drain_only=drain)
    assert report["ready_dates"] == 1
    assert report["planned_dates"] == (2 if drain else 1)
    assert calls == ["existing"]
    state = ExportState(tmp_path / "exports.sqlite")
    assert state.get("existing")["status"] == "cleaned"
    state.close()
