"""Exercise extension publication, joins, masked learning and frozen provenance."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine.config import Config
from torch import nn
from torch.nn import functional as F

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.datasets.collate import hydro_collate
from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.engine.layer_decay_constructor import olmoearth_layer_id
from hydrostate.engine.visualization_hook import HydroVisualizationHook, _cpu_sample
from hydrostate.evaluation.hydro_metrics import HydroStateMetric
from hydrostate.models import AuxiliaryInputFusion
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import prepare_supervision, supervision_values
from hydrostate.normalization import load_target_statistics
from hydrostate.registry import MODELS
from hydrostate.storage import SampleShardWriter, VariablePackWriter
from hydrostate.legacy.schema import LegacySchema as Schema
from hydrostate.storage.packs import VariablePack
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks, task_config
from tools.legacy.data.build_variable_pack import build_pack
from hydrostate.legacy.statistics import compute_statistics


@MODELS.register_module(force=True)
class PackTestEncoder(nn.Module):
    def __init__(self, multilevel=False):
        super().__init__()
        self.multilevel = multilevel

    def forward(self, inputs):
        value = F.adaptive_avg_pool2d(inputs["s1"][:, 0], (3, 3))
        return (value, value, value) if self.multilevel else value


@pytest.fixture
def library(tmp_path):
    schema = Schema(time_steps=1, height=12, width=12)
    writer = SampleShardWriter(str(tmp_path / "inputs.zarr"), str(tmp_path / "labels.zarr"),
                               4, schema=schema)
    for i in range(4):
        inputs = {k: np.ones(v.tail_shape, dtype=v.dtype) for k, v in schema.inputs.items()}
        targets = {k: np.zeros(v.tail_shape, dtype=v.dtype) for k, v in schema.targets.items()}
        for name in ("water", "soil_moisture", "et"):
            targets[name][:] = (i + 1) / 10
            targets[f"{name}_valid"][:] = True
        writer.write(inputs, targets)
        native = writer.label_group.require_group(f"et_native/{i}")
        native.create_dataset("weights", data=np.full((1, 4, 4), 1 / 16, dtype="float32"))
        native.create_dataset("value", data=np.array([i + 1], dtype="float32"))
        native.create_dataset("valid", data=np.ones(1, dtype=bool))
    writer.finalize()
    table = pd.DataFrame(dict(
        sample_id=["a", "b", "c", "d"], split=["train"] * 3 + ["validation"],
        input_shard=["inputs.zarr"] * 4, input_row=range(4),
        label_shard=["labels.zarr"] * 4, label_row=range(4),
        target_time=["2020-01-01"] * 4, crs=["EPSG:6933"] * 4,
        transform=[[10, 0, i * 120, 0, -10, 0] for i in range(4)],
    ))
    manifest = tmp_path / "samples.parquet"
    table.to_parquet(manifest, index=False)
    return manifest, table


def publish_extensions(manifest):
    root = manifest.parent
    with VariablePackWriter(
        root / "terrain_v1", manifest, kind="input", source="offline fixture", version="1",
        variables=dict(elevation=dict(shape=[1, 12, 12], units="m", temporal="static",
                                      grid="input")), shard_size=2,
    ) as writer:
        for i, sid in enumerate(("b", "a", "d")):
            writer.write(sid, dict(elevation=np.full((1, 12, 12), 100 * (i + 1)),
                                   elevation_valid=np.ones((1, 12, 12), dtype=bool)))
        writer.finalize()
    with VariablePackWriter(
        root / "temperature_v1", manifest, kind="label", source="offline fixture", version="1",
        variables=dict(temperature=dict(shape=[1, 4, 4], units="K", grid="output")),
    ) as writer:
        for i, sid in enumerate(("b", "a", "d")):
            value = np.arange(16, dtype="float32").reshape(1, 4, 4) + 280 + 10 * i
            valid = np.ones_like(value, dtype=bool)
            valid[..., 0, 0] = False
            value[..., 0, 0] = np.nan
            writer.write(sid, dict(temperature=value, temperature_valid=valid))
        writer.finalize()
    return dict(
        tasks=task_config(LEGACY_TASKS) | {"temperature": dict(support="pixel", grid="output")},
        input_packs=[dict(path="terrain_v1", variables=["elevation"])],
        label_packs=[dict(path="temperature_v1", variables=["temperature"])],
    )


def test_join_is_by_id_missing_rows_are_masked_and_output_labels_flip(library):
    manifest, table = library
    options = publish_extensions(manifest)
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    a, b, c = [dataset[i] for i in range(3)]
    assert a["inputs"]["elevation"].mean() == 200
    assert b["inputs"]["elevation"].mean() == 100
    assert not c["inputs"]["elevation_valid"].any()
    assert not c["data_samples"]["temperature_valid"].any()
    assert torch.isfinite(a["data_samples"]["temperature"]).all()
    original = a["data_samples"]["temperature"].clone()
    dataset.training, dataset.flip_probability = True, 1.0
    dataset._augment(a)
    torch.testing.assert_close(a["data_samples"]["temperature"], original.flip((-1, -2)))
    # Reordering the base table leaves the extension association intact.
    table.iloc[::-1].to_parquet(manifest, index=False)
    reordered = HydroStateZarrDataset(str(manifest), "train", **options)
    assert reordered[2]["inputs"]["elevation"].mean() == 200
    assert json.loads((manifest.parent / "terrain_v1/index.parquet").with_name(
        "metadata.json").read_text())["num_samples"] == 3


def test_publication_refuses_overwrite_and_rejects_wrong_identity(library):
    manifest, table = library
    options = publish_extensions(manifest)
    with pytest.raises(FileExistsError):
        VariablePackWriter(manifest.parent / "terrain_v1", manifest, kind="input", variables={},
                           source="fixture", version="1")
    table.loc[0, "split"] = "validation"
    table.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="identity/split/grid/time"):
        HydroStateZarrDataset(str(manifest), "train", **options)


def test_failed_import_is_not_published_and_rejects_bad_ids_masks(library):
    manifest, _ = library
    destination = manifest.parent / "failed"
    values = dict(x=np.ones((1, 12, 12)), x_valid=np.ones((1, 12, 12)))
    with VariablePackWriter(destination, manifest, kind="input", source="fixture", version="1",
                            variables=dict(x=dict(shape=[1, 12, 12], temporal="static",
                                                  units="1", grid="input"))) as writer:
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            writer.write("unknown", values)
        values["x_valid"][0, 0, 0] = 2
        with pytest.raises(ValueError, match="masks must be"):
            writer.write("a", values)
        values["x_valid"][0, 0, 0] = 1
        writer.write("a", values)
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            writer.write("a", values)
    assert not destination.exists()
    assert not list(destination.parent.glob(".failed.*.partial"))


@pytest.mark.parametrize("kind", ["HydroStateModel", "StagedHydroStateModel",
                                  "MultiLevelHydroStateModel"])
def test_new_input_and_fourth_target_train_and_evaluate(library, kind):
    manifest, _ = library
    options = publish_extensions(manifest)
    statistics = compute_statistics(manifest, manifest.parent, num_workers=0, **options)
    assert statistics["tasks"]["temperature"]["count"] == 30
    assert statistics["tasks"]["temperature"]["mean"] == 293
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    preprocessor = HydroDataPreprocessor(tasks=options["tasks"])
    batch = preprocessor(hydro_collate([dataset[i] for i in range(3)]))
    decoder_type = dict(HydroStateModel="SharedUNetDecoder",
                        StagedHydroStateModel="StagedTaskDecoder",
                        MultiLevelHydroStateModel="MultiLevelFusionDecoder")[kind]
    model = MODELS.build(dict(
        type=kind, tasks=options["tasks"], target_statistics=statistics,
        encoder=dict(type="PackTestEncoder", multilevel=kind == "MultiLevelHydroStateModel"),
        decoder=dict(type=decoder_type, in_channels=2, channels=[8, 4, 4]),
        heads={name: dict(type="RegressionHead", in_channels=4) for name in options["tasks"]},
        loss=dict(type="UncertaintyWeightedNLL"),
        input_fusion=dict(type="AuxiliaryInputFusion", feature_channels=2,
                          inputs=dict(elevation=dict(channels=1, mean=[0], std=[1000]))),
    ))
    losses = model(**batch, mode="loss")
    assert losses["valid_count_temperature"] == 30
    total = sum(value for name, value in losses.items() if name.startswith("loss_"))
    total.backward()
    assert torch.isfinite(total)
    assert model.heads["temperature"].layers[-1].weight.grad.abs().sum() > 0
    assert model.input_fusion.branches["elevation"].layers[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        predictions = model(**batch, mode="predict")
    metric = HydroStateMetric(statistics, tasks=options["tasks"])
    metric.process(None, predictions)
    metrics = metric.compute_metrics(metric.results)
    assert np.isfinite(metrics["temperature_rmse"])
    assert set(predictions[0]) >= set(options["tasks"])


def test_stats_and_config_pin_pack_versions_and_supports(library):
    manifest, _ = library
    options = publish_extensions(manifest)
    report = compute_statistics(manifest, manifest.parent, num_workers=0, **options)
    cfg = Config(dict(
        **options,
        model=dict(target_statistics=report, input_fusion=dict(inputs=dict(elevation={}))),
        train_dataloader=dict(dataset=dict(
            type="HydroStateZarrDataset", manifest=str(manifest), split="train")),
        val_evaluator=dict(type="HydroStateMetric"),
    ))
    configure_target_statistics(cfg, training=True)
    assert cfg.model.tasks == cfg.train_dataloader.dataset.tasks == cfg.val_evaluator.tasks
    changed = copy.deepcopy(options["tasks"])
    changed["temperature"] = dict(support="window_mean", grid="scalar")
    with pytest.raises(ValueError, match="task supports"):
        load_target_statistics(report, changed)
    cfg.input_packs = []
    cfg.train_dataloader.dataset.input_packs = []
    cfg.model.input_fusion.inputs = {}
    with pytest.raises(ValueError, match="manifest differs"):
        configure_target_statistics(cfg, training=True)


def test_auxiliary_sequence_mean_ignores_missing_values_and_all_missing_is_identity():
    layer = AuxiliaryInputFusion(feature_channels=2, inputs=dict(
        rain=dict(channels=1, mean=[0], std=[2], temporal="mean")))
    features = torch.randn(2, 2, 2, 2)
    values = torch.ones(2, 3, 1, 4, 4)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[0, 1:] = False
    valid[1] = False
    first = layer(features, dict(rain=values, rain_valid=valid))
    values[~valid] = float("nan")
    second = layer(features, dict(rain=values, rain_valid=valid))
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(second[1], features[1], rtol=0, atol=0)
    assert olmoearth_layer_id("input_fusion.branches.rain.layers.2.weight", 12) == 13


def test_custom_scalar_task_uses_window_support_and_no_legacy_station_slot():
    tasks = dict(runoff=dict(support="window_mean", grid="scalar"))
    prediction = torch.tensor([[[[1., 3.], [5., 99.]]]], requires_grad=True)
    samples = dict(runoff=torch.tensor([[[[2.]]]]), runoff_valid=torch.ones(1, 1, 1, 1).bool(),
                   prediction_valid=torch.tensor([[[[True, True], [True, False]]]]),
                   site_value=torch.ones(1, 3), site_valid=torch.ones(1, 3).bool(),
                   site_footprint=torch.ones(1, 1, 2, 2))
    plan = prepare_supervision(dict(runoff=prediction), samples, tasks=tasks)["runoff"]
    assert plan.project(prediction).item() == 3
    assert plan.count == 1
    assert plan.site_footprint is None
    projected, _, _ = supervision_values("runoff", prediction, samples, tasks=tasks)
    torch.testing.assert_close(plan.project(prediction), projected)
    projected.sum().backward()
    assert prediction.grad[0, 0, 1, 1] == 0
    with pytest.raises(ValueError, match="declare"):
        resolve_tasks(dict(runoff={}))


def test_npz_import_and_example_configuration(library):
    manifest, _ = library
    np.savez(manifest.parent / "a.npz", rain=np.ones((2, 1, 12, 12)),
             rain_valid=np.ones((2, 1, 12, 12), bool), rain_time=np.array([100, 200]))
    records = manifest.parent / "records.parquet"
    pd.DataFrame(dict(sample_id=["a"], npz_path=["a.npz"])).to_parquet(records)
    output = build_pack(manifest, records, dict(
        kind="input", source="fixture", version="1",
        variables=dict(rain=dict(shape=[2, 1, 12, 12], units="mm", temporal="sequence",
                                 grid="input"))), manifest.parent / "rain_v1")
    pack = VariablePack(output, pd.read_parquet(manifest), kind="input")
    assert pack.variables["rain"]["temporal"] == "sequence"
    cfg = Config.fromfile("configs/experiments/legacy/hydrostate_extended.py")
    configure_data_contract(cfg)
    assert set(cfg.model.heads) == set(cfg.model.tasks)
    assert cfg.train_dataloader.dataset.label_packs == cfg.label_packs


def test_label_source_override_does_not_fall_back_to_old_labels(library):
    manifest, _ = library
    with VariablePackWriter(
        manifest.parent / "sm_v2", manifest, kind="label", source="new source", version="2",
        variables=dict(soil_moisture=dict(shape=[1, 1, 1], units="m3/m3", grid="scalar")),
    ) as writer:
        writer.write("a", dict(soil_moisture=np.full((1, 1, 1), 0.8),
                               soil_moisture_valid=np.ones((1, 1, 1), bool)))
        writer.finalize()
    dataset = HydroStateZarrDataset(str(manifest), "train", label_packs=[dict(path="sm_v2")])
    assert dataset[0]["data_samples"]["soil_moisture"].item() == pytest.approx(0.8)
    assert not dataset[1]["data_samples"]["soil_moisture_valid"].any()
    with pytest.raises(ValueError, match="Multiple label packs"):
        HydroStateZarrDataset(str(manifest), "train", label_packs=[dict(path="sm_v2")] * 2)


def test_label_times_roundtrip_missing_rows_and_temporal_data_contract(library):
    from hydrostate.data.target_times import epoch_seconds

    manifest, table = library
    table["labels"] = json.dumps(dict(dw={"system:time_start": 1577870000000},
                                     sm={"system:time_start": 1577880000000},
                                     pml=dict(interval_start="2020-01-01",
                                              interval_end="2020-01-09")))
    table.to_parquet(manifest, index=False)
    with VariablePackWriter(
        manifest.parent / "lst_v1", manifest, kind="label", source="fixture", version="1",
        variables=dict(lst=dict(shape=[1, 4, 4], units="K", grid="output", time_support="instant")),
    ) as writer:
        value = dict(lst=np.arange(16).reshape(1, 4, 4) + 280,
                     lst_valid=np.ones((1, 4, 4), bool), lst_time=np.array([1577600000] * 2))
        with pytest.raises(ValueError, match="time"):
            writer.write("a", dict(value, lst_time=np.array([0, 0])))
        with pytest.raises(ValueError, match="time"):
            writer.write("a", dict(value, lst_time=np.array([1577600000, 1577600001])))
        writer.write("a", value)
        writer.finalize()
    options = dict(tasks=task_config(LEGACY_TASKS) | {"lst": dict(support="pixel", grid="output")},
                   label_packs=[dict(path="lst_v1")], target_time_queries=True)
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    a, b = dataset[0], dataset[1]
    assert a["inputs"]["target_time"]["lst"].tolist() == [1577600000] * 2
    assert a["inputs"]["target_time"]["soil_moisture"].tolist() == [1577880000] * 2
    assert a["inputs"]["target_time"]["et"].diff().item() == 8 * 86400
    assert b["inputs"]["target_time"]["lst"].tolist() == [epoch_seconds("2020-01-01")] * 2
    assert not b["data_samples"]["lst_valid"].any()
    batch = HydroDataPreprocessor(tasks=options["tasks"])(hydro_collate([a, b]))
    assert batch["inputs"]["target_time"]["lst"].shape == (2, 2)
    assert batch["inputs"]["target_time"]["lst"].dtype == torch.long
    statistics = compute_statistics(manifest, manifest.parent, num_workers=0, **options)
    assert statistics["data_contract"]["target_time_queries"] == "observation_intervals_v1"
    old = compute_statistics(manifest, manifest.parent, num_workers=0,
                             **dict(options, target_time_queries=False))
    assert statistics["training_fingerprint"] != old["training_fingerprint"]


def test_temporal_queries_require_replacement_labels_own_times(library):
    manifest, table = library
    options = publish_extensions(manifest)
    dataset = HydroStateZarrDataset(str(manifest), "train", tasks={
        "temperature": dict(support="pixel", grid="output")},
        label_packs=options["label_packs"], target_time_queries=True)
    with pytest.raises(ValueError, match="temperature: valid labels require actual observation"):
        dataset[0]


def test_arbitrary_new_variable_can_supply_an_interval_query(library):
    manifest, _ = library
    with VariablePackWriter(
        manifest.parent / "biomass", manifest, kind="label", source="fixture", version="1",
        variables=dict(biomass=dict(shape=[1, 1, 1], units="kg/m2", grid="scalar",
                                    time_support="interval")),
    ) as writer:
        writer.write("a", dict(biomass=np.ones((1, 1, 1)), biomass_valid=np.ones((1, 1, 1), bool),
                               biomass_time=np.array([1577836800, 1577836800 + 30 * 86400])))
        writer.finalize()
    dataset = HydroStateZarrDataset(str(manifest), "train", tasks={
        "biomass": dict(support="window_mean", grid="scalar")},
        label_packs=[dict(path="biomass")], target_time_queries=True)
    assert dataset[0]["inputs"]["target_time"]["biomass"].diff().item() == 30 * 86400
    assert not dataset[1]["data_samples"]["biomass_valid"].any()


def test_support_validator_preserves_all_real_input_masks_and_times(library):
    from tools.legacy.data.validate_gee_lst import SupportDataset

    manifest, _ = library
    reference = HydroStateZarrDataset(str(manifest), "train")[0]
    support = SupportDataset(str(manifest), "train")[0]
    for key in ("s1_valid", "s2_valid", "landsat_valid", "sensor_valid", "observation_time"):
        torch.testing.assert_close(support["inputs"][key], reference["inputs"][key])
    actual = HydroDataPreprocessor()(hydro_collate([support]))["data_samples"]
    expected = HydroDataPreprocessor()(hydro_collate([reference]))["data_samples"]
    for key in actual:
        if torch.is_tensor(actual[key]):
            torch.testing.assert_close(actual[key], expected[key])


def test_temperature_only_statistics_batch_and_visualization(library, tmp_path):
    manifest, _ = library
    options = publish_extensions(manifest)
    options["tasks"] = dict(temperature=dict(support="pixel", grid="output"))
    options["input_packs"] = []
    statistics = compute_statistics(manifest, manifest.parent, num_workers=0, **options)
    assert set(statistics["tasks"]) == {"temperature"}
    dataset = HydroStateZarrDataset(str(manifest), "train", **options)
    batch = HydroDataPreprocessor(tasks=options["tasks"])(hydro_collate([dataset[0]]))
    assert "et_native_value" not in batch["data_samples"]
    sample = batch["data_samples"]
    values = sample["temperature"][0]
    prediction = dict(temperature=values, supervised_temperature=values,
                      target_temperature=values, valid_temperature=sample["temperature_valid"][0],
                      prediction_valid=sample["prediction_valid"][0],
                      metainfo=dict(sample_id="a"))
    hook = HydroVisualizationHook(tasks=options["tasks"])
    hook._label = "test"
    hook._draw(_cpu_sample(prediction), tmp_path / "temperature.png")
    assert (tmp_path / "temperature.png").stat().st_size > 0
