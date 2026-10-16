"""Scientific support, quality masks, resumability and ecosystem model integration."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine.config import Config

from hydrostate.cli import configure_data_contract
from hydrostate.data.gee_labels import COVER_NAMES, containing_interval, quality_masks
from hydrostate.data.native_support import NativeSupport
from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset, hydro_collate
from hydrostate.engine.visualization_hook import comparison_maps
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import prepare_supervision, supervision_values
from hydrostate.registry import MODELS
from hydrostate.storage import VariablePackWriter
from tests import test_variable_packs as pack_fixtures
from tools.data.compute_target_statistics import compute_statistics
from tools.data.extend_gee_labels import Checkpoint, select_rows
from tools.data.validate_gee_labels import validate_training_support


@pytest.mark.parametrize("stations", ["absent", "masked", "valid"])
def test_validation_matches_product_and_station_support(stations):
    from hydrostate.tasks import task_config

    specs = task_config() | dict(lai=dict(support="native", grid="native"),
                                lst=dict(support="pixel", grid="output"))
    support = torch.ones(2, 1, 2, 2, dtype=torch.bool)
    support[0, 0, 1, 1] = False
    samples = dict(prediction_valid=support)
    for name in ("water", "lst"):
        samples[name] = torch.full((2, 1, 2, 2), .5)
        samples[name + "_valid"] = torch.ones_like(support)
    samples["soil_moisture"] = torch.full((2, 1, 1, 1), .5)
    samples["soil_moisture_valid"] = torch.ones(2, 1, 1, 1, dtype=torch.bool)
    samples["et_native_weights"] = torch.full((2, 1, 2, 2), .25)
    samples["et_native_value"] = torch.full((2, 1), .5)
    samples["et_native_valid"] = torch.ones(2, 1, dtype=torch.bool)
    weights = torch.zeros(2, 2, 2, 2)
    weights[:, 0, :, 0] = .5
    weights[:, 1, :, 1] = .5
    samples["lai_native_weights"] = weights
    samples["lai_native_value"] = torch.full((2, 2), .5)
    samples["lai_native_valid"] = torch.ones(2, 2, dtype=torch.bool)
    if stations != "absent":
        samples["site_value"] = torch.full((2, 3), .25)
        samples["site_valid"] = torch.full((2, 3), stations == "valid", dtype=torch.bool)
        samples["site_footprint"] = torch.zeros(2, 1, 2, 2)
        if stations == "valid":
            samples["site_footprint"][0, 0, 0, 0] = 1
            samples["site_footprint"][1] = .25
    outputs = {name: torch.arange(1., 9.).reshape(2, 1, 2, 2).requires_grad_()
               for name in specs}
    loss, counts = validate_training_support(outputs, samples, specs)
    extra = 2 if stations == "valid" else 0
    assert counts == dict(water=7 + extra, soil_moisture=2 + extra,
                          et=1 + extra, lai=3, lst=7)
    loss.backward()
    assert torch.isfinite(loss)
    for value in outputs.values():
        assert torch.isfinite(value.grad).all()
        assert value.grad[0, 0, 1, 1] == 0


def test_native_geometry_complete_cells_and_exact_area_means():
    sample = dict(lat=0, lon=0, crs="EPSG:6933", transform=[10, 0, 0, 0, -10, 120])
    support = NativeSupport(sample, shape=(12, 12))
    projection = dict(crs="EPSG:6933", transform=[60, 0, 0, 0, -60, 120])
    raster = np.repeat(np.repeat(np.array([[2., 4.], [6., 8.]]), 6, 0), 6, 1)
    result = support.targets(raster, np.ones((12, 12), bool), projection, capacity=4)
    np.testing.assert_allclose(result["value"], [2, 4, 6, 8])
    np.testing.assert_allclose(result["weights"].sum(axis=(1, 2)), 1)
    expected = np.zeros((4, 4, 4))
    for i in range(4):
        expected[i, i // 2 * 2:i // 2 * 2 + 2, i % 2 * 2:i % 2 * 2 + 2] = .25
    np.testing.assert_allclose(result["weights"], expected, atol=1e-6)
    # An intersecting coarse cell extending outside the window is never normalized inwards.
    oversized = dict(crs="EPSG:6933", transform=[120, 0, -30, 0, -120, 150])
    assert not support.targets(raster, np.ones((12, 12), bool), oversized)["valid"].any()
    with pytest.raises(ValueError, match="exceed capacity"):
        support.targets(raster, np.ones((12, 12), bool), projection, capacity=1)


def test_composites_stop_at_year_boundary_and_qa_does_not_reject_saturated_lai():
    assert containing_interval("2020-12-26", 8) == ("2020-12-26", "2021-01-01")
    assert containing_interval("2019-12-31", 4) == ("2019-12-31", "2020-01-01")
    data = {k: np.ones(5) for k in ("dw_valid", "lai_valid", "gpp_valid",
                                     "albedo_valid", "lst_valid")}
    data.update(dw_class=np.array([0, 8, 9, 1, 2]), lai_raw=np.full(5, 20.),
                lai_qc=np.array([0, 32, 64, 8, 0]), lai_extra=np.array([0, 0, 0, 0, 4]),
                gpp_raw=np.array([100, 0, -1, 65535, 100]),
                albedo_qc=np.array([0, 1, 0, 0, 0]),
                albedo_bsa_raw=np.array([100, 100, 32767, 100, 100]),
                albedo_wsa_raw=np.array([120, 120, 120, -1, 120]),
                lst_raw=np.full(5, 45000.), lst_qa=np.array([0, 8, 16, 32, 0]),
                lst_uncertainty_raw=np.array([100, 100, 100, 100, 201]))
    masks = quality_masks(data, lst_max_uncertainty_k=2)
    np.testing.assert_array_equal(masks["lai"], [True, True, False, False, False])
    np.testing.assert_array_equal(masks["albedo"], [True, False, False, False, True])
    np.testing.assert_array_equal(masks["gpp"], [True, True, False, False, True])
    np.testing.assert_array_equal(masks["lst"], [True, False, False, True, False])


def native_pack(manifest):
    specs = dict(lai=dict(grid="native", shape=[4], spatial_shape=[4, 4], units="m2/m2"))
    with VariablePackWriter(manifest.parent / "lai_v1", manifest, kind="label", source="test",
                            version="1", variables=specs, shard_size=2,
                            chunk_samples=2) as writer:
        for j, sid in enumerate(("a", "b", "d")):
            weights = np.zeros((4, 4, 4), np.float32)
            weights[0, :2, :2] = .25
            weights[1, 2:, 2:] = .25
            writer.write(sid, dict(lai=np.array([j + 1, j + 2, 0, 0], np.float32),
                                    lai_valid=np.array([True, j != 1, False, False]),
                                    lai_weights=weights), provenance={"asset_id": f"test/{sid}"})
        writer.finalize()
    return dict(tasks=dict(lai=dict(support="native", grid="native")),
                label_packs=[dict(path="lai_v1")])


@pytest.fixture
def library(tmp_path):
    return pack_fixtures.library.__wrapped__(tmp_path)


def test_native_pack_join_padding_augmentation_stats_and_unobserved_support(library):
    manifest, _ = library
    options = native_pack(manifest)
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    assert dataset[0]["data_samples"]["lai_native_value"].tolist() == [1, 2]
    assert dataset[1]["data_samples"]["lai_native_value"].tolist() == [2]
    assert not dataset[2]["data_samples"]["lai_native_valid"].any()
    item = dataset[0]
    original = item["data_samples"]["lai_native_weights"].clone()
    dataset.training, dataset.flip_probability = True, 1
    dataset._augment(item)
    torch.testing.assert_close(item["data_samples"]["lai_native_weights"], original.flip((-1, -2)))
    dataset.training = False
    batch = HydroDataPreprocessor(tasks=options["tasks"])(
        hydro_collate([dataset[0], dataset[1], dataset[2]]))
    samples = batch["data_samples"]
    samples["prediction_valid"][0, 0, 0, 0] = False
    prediction = torch.arange(48.).reshape(3, 1, 4, 4).requires_grad_()
    pooled, target, valid = supervision_values("lai", prediction, samples, tasks=options["tasks"])
    assert valid.tolist() == [[False, True], [True, False], [False, False]]
    prepared = prepare_supervision(dict(lai=prediction), samples, tasks=options["tasks"])["lai"]
    torch.testing.assert_close(prepared.project(prediction), pooled)
    assert prepared.count == 2
    ((pooled - target)[valid] ** 2).sum().backward()
    assert prediction.grad[0, 0, 0, 0] == 0
    stats = compute_statistics(manifest, manifest.parent, num_workers=0, **options)
    assert stats["tasks"]["lai"]["count"] == 3
    assert stats["tasks"]["lai"]["mean"] == pytest.approx(5 / 3)
    sample = dict(lai=prediction[0].detach().numpy(),
                  prediction_valid=samples["prediction_valid"][0].numpy(),
                  lai_native_weights=samples["lai_native_weights"][0].numpy(),
                  supervised_lai=pooled[0].detach().numpy(), target_lai=target[0].numpy(),
                  valid_lai=valid[0].numpy())
    assert all(value.shape == (4, 4) for value in comparison_maps(sample, "lai"))


@pytest.mark.parametrize("kind", ["HydroStateModel", "StagedHydroStateModel",
                                  "MultiLevelHydroStateModel"])
def test_fraction_outputs_and_native_losses_for_all_model_types(library, kind):
    manifest, _ = library
    options = native_pack(manifest)
    options["tasks"].update(water=dict(support="pixel", grid="input"),
                            trees=dict(support="pixel", grid="input"))
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    batch = HydroDataPreprocessor(tasks=options["tasks"])(hydro_collate([dataset[0], dataset[1]]))
    batch["data_samples"]["trees"] = 1 - batch["data_samples"]["water"]
    batch["data_samples"]["trees_valid"] = batch["data_samples"]["water_valid"]
    stats = compute_statistics(manifest, manifest.parent, num_workers=0,
                               **native_pack_options())
    from hydrostate.tasks import task_config

    stats["task_specs"] = task_config(options["tasks"])
    stats["tasks"].update(water=dict(count=32, mean=.15, std=.05),
                         trees=dict(count=32, mean=.85, std=.05))
    decoder = dict(HydroStateModel="SharedUNetDecoder", StagedHydroStateModel="StagedTaskDecoder",
                   MultiLevelHydroStateModel="MultiLevelFusionDecoder")[kind]
    model = MODELS.build(dict(
        type=kind, tasks=options["tasks"], target_statistics=stats,
        encoder=dict(type="PackTestEncoder", multilevel=kind == "MultiLevelHydroStateModel"),
        decoder=dict(type=decoder, in_channels=2, channels=[8, 4, 4]),
        heads={name: dict(type="RegressionHead", in_channels=4) for name in options["tasks"]},
        fraction_groups=[["water", "trees"]], loss=dict(type="UncertaintyWeightedNLL")))
    outputs = model(batch["inputs"], mode="tensor")
    torch.testing.assert_close(outputs["water"] + outputs["trees"],
                               torch.ones_like(outputs["water"]))
    losses = model(**batch, mode="loss")
    loss = sum(v for k, v in losses.items() if k.startswith("loss_"))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.heads["lai"].layers[-1].weight.grad.abs().sum() > 0
    predicted = model(**batch, mode="predict")
    assert "lai_native_weights" in predicted[0]


def native_pack_options():
    return dict(tasks=dict(lai=dict(support="native", grid="native")),
                label_packs=[dict(path="lai_v1")])


def test_checkpoint_resume_contract_and_ecosystem_configuration(tmp_path):
    path = tmp_path / "state.sqlite3"
    cache = Checkpoint(path, dict(version="1"))
    cache.fail("a", "network error")
    assert not cache.completed()
    cache.save("a", dict(x=np.array([3.])), {"scenes": {"lst": None}})
    cache.close()
    cache = Checkpoint(path, dict(version="1"))
    values, provenance = cache.read("a")
    assert cache.completed() == {"a"}
    assert values["x"].item() == 3
    assert provenance["scenes"]["lst"] is None
    cache.close()
    with pytest.raises(ValueError, match="parameters/manifest differ"):
        Checkpoint(path, dict(version="2"))
    cfg = Config.fromfile("configs/hydrostate_ecosystem.py")
    configure_data_contract(cfg)
    assert len(cfg.tasks) == 16
    assert set(cfg.tasks) == set(cfg.model.heads)
    assert cfg.model.fraction_groups == [list(COVER_NAMES)]
    assert all(cfg.model.heads[name].activation == "identity" for name in COVER_NAMES)
    assert cfg.label_packs[0].max_uncertainty.lst == 2
    table = pd.DataFrame([dict(sample_id=str(i), target_time=f"{2019+i}-01-01", lat=i*10,
                               split="train", frequency_bin=i) for i in range(6)])
    a, b = select_rows(table, 3), select_rows(table.iloc[::-1], 3)
    assert json.dumps(a) == json.dumps(b)
