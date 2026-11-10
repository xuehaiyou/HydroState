import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from mmengine.config import Config
from mmengine.registry import init_default_scope
from test_dataset import make_dataset
from test_model import statistics

from hydrostate.cli import configure_data_contract, configure_target_statistics
from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.site.archives import sha256
from hydrostate.data.site.contract import SiteSchema
from hydrostate.data.site.statistics import cohort_statistics
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.joint_dataset import JointRetrievalDataset
from hydrostate.evaluation.joint_metrics import JointHydroStateMetric
from hydrostate.models.site_supervision import SiteObjective
from hydrostate.registry import MODELS
from hydrostate.storage import SampleShardWriter


@pytest.fixture
def joint(tmp_path):
    product, site, split, proof = [tmp_path / s for s in ("product", "site", "v3", "proof")]
    for path in (product, site, split, proof):
        path.mkdir()
    manifest = make_dataset(product)
    ptable = pd.read_parquet(manifest)
    schema = SiteSchema()
    config = dict(stage="fixture")
    (site / "collection.json").write_text(
        json.dumps(dict(config, version=schema.version, input_profile=ptable.input_profile.iloc[0]))
    )
    rows = []
    for i, row in ptable.iterrows():
        inputs = zarr.open_group(str(product / row.input_path), mode="r")
        labels = {k: np.zeros(s.tail_shape, s.dtype) for k, s in schema.targets.items()}
        for task in ("sm", "et"):
            labels[f"site_{task}_native_value"][0] = 0.2 + 0.1 * i
            labels[f"site_{task}_native_valid"][0] = 1
            labels[f"site_{task}_native_weights"][0, 20, 20] = 1
        writer = SampleShardWriter(
            str(site / row.input_path),
            str(site / row.label_path),
            1,
            schema,
            attrs=dict(config_sha256=fingerprint(config), input_grids=inputs.attrs["input_grids"]),
        )
        writer.write({k: inputs[k][0] for k in schema.inputs}, labels)
        writer.finalize()
        rows.append(
            dict(
                row,
                sample_id=f"s{i}",
                candidate_id=f"s{i}",
                version=schema.version,
                observation_capacity=1,
                physical_group_id=f"g{i}",
                spatial_group_id=f"g{i}",
                valid_sm=1,
                valid_et=1,
                cohort=0,
                et_support="static_210m",
            )
        )
    table = pd.DataFrame(rows)
    table.to_parquet(site / "samples.parquet", index=False)
    table.to_parquet(split / "cohort_0.parquet", index=False)
    table.to_parquet(split / "inventory.parquet", index=False)
    ptable[["sample_id", "split"]].to_parquet(proof / "selection.parquet", index=False)
    (proof / "product_support.json").write_text(json.dumps(dict(manifest_sha256=sha256(manifest))))
    table.assign(product_overlap_count=0, split_reason="seeded").to_parquet(
        proof / "station_splits.parquet", index=False
    )
    report = dict(
        version="site_frequency_task_spatial_split_v3",
        statistics_status="complete_train_only",
        task_coverage=[],
        source_manifests={str(site): sha256(site / "samples.parquet")},
        source_splits_sha256=sha256(proof / "station_splits.parquet"),
        joint_product_selection=str(proof / "selection.parquet"),
        joint_product_selection_sha256=sha256(proof / "selection.parquet"),
    )
    (split / "split_report.json").write_text(json.dumps(report))
    datasets = [
        dict(type="SiteDataset", manifest=str(split / "cohort_0.parquet"), data_root=str(site))
    ]
    (split / "datasets.json").write_text(json.dumps(dict(datasets=datasets)))
    stats = cohort_statistics(datasets)
    (split / "site_target_stats.json").write_text(json.dumps(stats))
    dataset = JointRetrievalDataset(manifest, split, "train", data_root=product)
    return dataset, stats, report


def test_mixed_batch_masks_sampling_and_statistic_mismatch(joint):
    dataset, stats, report = joint
    assert len(dataset) == 4
    weights = dataset.sampling_weights()
    torch.testing.assert_close(weights, torch.full((4,), 0.25, dtype=torch.double))
    batch = hydro_collate([dataset[0], dataset[2]])
    assert not batch["data_samples"]["site_sm_native_valid"][0].any()
    assert not batch["data_samples"]["soil_moisture_valid"][1].any()
    assert batch["data_samples"]["site_sm_native_valid"][1].all()
    dataset.validate_site_statistics(stats)
    broken = copy.deepcopy(stats)
    broken["cohorts"][0]["training_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="mismatch"):
        dataset.validate_site_statistics(broken)
    with pytest.raises(ValueError, match="moments disagree"):
        broken = copy.deepcopy(stats)
        broken["observations"]["site_sm"]["mean"] += 1
        SiteObjective(broken)
    # The exclusion list may remove evaluation records, never product training records.
    selection = pd.read_parquet(report["joint_product_selection"]).iloc[:1]
    selection.to_parquet(report["joint_product_selection"], index=False)
    report["joint_product_selection_sha256"] = sha256(report["joint_product_selection"])
    (dataset.site_root / "split_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training set changed"):
        JointRetrievalDataset(dataset.manifest, dataset.site_root, "train", dataset.product.root)


@pytest.mark.parametrize("workers", [0, 2])
def test_mixed_loader_mask_dtypes_do_not_depend_on_first_source(joint, workers):
    from torch.utils.data import DataLoader

    dataset, _, _ = joint
    loader = DataLoader(dataset, batch_size=2, sampler=[2, 0, 0, 2], num_workers=workers,
                        collate_fn=hydro_collate)
    for batch in loader:
        masks = batch["data_samples"]
        for name, values in masks.items():
            if name.endswith("_valid"):
                assert values.dtype == torch.bool, name
        assert masks["site_sm_native_valid"].sum() == 1
        assert masks["soil_moisture_valid"].sum() == 1


def test_mixed_real_zarr_forward_backward_and_separate_metrics(joint):
    dataset, stats, _ = joint
    init_default_scope("hydrostate")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = Config.fromfile("configs/hydrostate_unet.py")
        configure_data_contract(cfg)
        cfg.model.encoder.channels = [8, 16, 32, 64]
        cfg.model.decoder.update(encoder_channels=[8, 16, 32, 64], channels=8)
        for head in cfg.model.heads.values():
            head.in_channels = 8
        cfg.model.target_statistics = statistics()
        cfg.model.site_supervision = dict(statistics=stats)
        model = MODELS.build(cfg.model)
        samples = [dataset[0], dataset[2]]
        for sample, year in zip(samples, (2018, 2022)):
            sample["data_samples"]["metainfo"]["year"] = year
        data = model.data_preprocessor(hydro_collate(samples), training=True)
        losses = model.loss(**data)
        assert losses["valid_count_site_sm"] == 1
        assert losses["valid_count_soil_moisture"] == 1
        total = sum(v for k, v in losses.items() if k.startswith("loss_"))
        assert torch.isfinite(total)
        total.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        predictions = model.predict(**data)
        assert [int(p["metainfo"]["year"]) for p in predictions] == [2018, 2022]
        metric = JointHydroStateMetric(statistics(), stats, cfg.tasks,
                                      comparison_years=(2017, 2024))
        metric.process({}, predictions)
        result = metric.compute_metrics(metric.results)
        assert result["site/site_sm_count"] == 1
        assert result["product/soil_moisture_count"] == 1
        assert result["site/frequency_0/site_sm_spatial_groups"] == 1
        assert np.isfinite(result["joint_score"])
        assert result["product/years_2017_2024/soil_moisture_count"] == 1
        assert result["site/years_2017_2024/site_sm_count"] == 1
    finally:
        torch.set_num_threads(previous_threads)


def test_joint_config_binds_statistics_and_checkpoint_score(joint):
    dataset, stats, _ = joint
    cfg = Config.fromfile("configs/hydrostate_joint.py")
    for name in ("train", "val", "test"):
        cfg[f"{name}_dataloader"].dataset.update(
            manifest=dataset.manifest,
            data_root=str(dataset.product.root),
            site_split_root=str(dataset.site_root),
        )
    from hydrostate.normalization import training_fingerprint

    product_stats = statistics()
    product_stats["training_fingerprint"] = training_fingerprint(
        pd.read_parquet(dataset.manifest), dataset.product.data_contract
    )
    cfg.model.target_statistics = product_stats
    cfg.model.site_supervision.statistics = stats
    configure_target_statistics(cfg, training=True)
    assert cfg.val_evaluator.site_statistics == stats
    assert cfg.default_hooks.checkpoint.save_best == "joint_score"
    assert cfg.val_evaluator.site_score_weight == pytest.approx(2 / 3)
    assert cfg.test_evaluator.site_score_weight == pytest.approx(2 / 3)
    assert cfg.model.type == "CoSAHydroStateModel"
    independent = Config.fromfile("configs/hydrostate_joint_independent.py")
    conditional = Config.fromfile("configs/hydrostate_joint.py")
    independent.model.router.enabled = True
    for key in ("model", "train_dataloader", "val_dataloader", "test_dataloader",
                "optim_wrapper", "param_scheduler", "train_cfg", "val_evaluator",
                "test_evaluator", "randomness"):
        assert independent[key] == conditional[key]


def test_long_station_series_does_not_gain_sampling_mass():
    class Table:
        def __init__(self, table):
            self.table = table

        def __len__(self):
            return len(self.table)

    dataset = object.__new__(JointRetrievalDataset)
    dataset.site_fraction = 0.5
    dataset.product = Table(pd.DataFrame(dict(sampling_group=[0, 0, 1])))
    dataset.sites = [
        Table(
            pd.DataFrame(
                [
                    dict(frequency_bin=0, spatial_group_id="long", valid_sm=1, valid_et=0)
                    for _ in range(4)
                ]
                + [
                    dict(frequency_bin=0, spatial_group_id="short", valid_sm=1, valid_et=0),
                    dict(frequency_bin=1, spatial_group_id="rare", valid_sm=0, valid_et=1),
                ]
            )
        )
    ]
    dataset.offsets = [0, 3, 9]
    weights = dataset.sampling_weights()
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3:7].sum() == pytest.approx(weights[7])
    assert weights[3:8].sum() == pytest.approx(weights[8])


def test_station_priority_scales_loss_without_changing_valid_count(joint):
    from hydrostate.models.site_supervision import station_values

    _, stats, _ = joint
    outputs = {
        name: torch.full((1, 1, 2, 2), 0.3, requires_grad=True)
        for name in ("soil_moisture", "water", "snow_ice", "land", "et_water", "et_ice", "et_land")
    }
    samples = {}
    for name in ("site_sm", "site_et"):
        samples[f"{name}_native_value"] = torch.tensor([[0.2]])
        samples[f"{name}_native_valid"] = torch.ones(1, 1, dtype=torch.bool)
        samples[f"{name}_native_weights"] = torch.full((1, 1, 2, 2), 0.25)
    ordinary = SiteObjective(stats)(outputs, samples)
    priority = SiteObjective(stats, dict(site_sm=2, site_et=2))(outputs, samples)
    for name in ("site_sm", "site_et"):
        torch.testing.assert_close(priority[f"loss_{name}"], 2 * ordinary[f"loss_{name}"])
        assert priority[f"valid_count_{name}"] == ordinary[f"valid_count_{name}"] == 1
    assert station_values(outputs, samples)["site_sm"][2].all()
    cfg = Config.fromfile("configs/hydrostate_joint.py")
    assert cfg.model.site_supervision.loss_weights == dict(site_sm=2, site_et=2)


def test_rare_frequency_and_extra_dates_do_not_amplify_spatial_group_mass():
    from types import SimpleNamespace

    dataset = object.__new__(JointRetrievalDataset)
    dataset.site_fraction = 0.5
    dataset.product = type("Products", (), {
        "table": pd.DataFrame(dict(sampling_group=[0, 1])), "__len__": lambda self: 2,
    })()
    rows = [dict(frequency_bin=frequency, spatial_group_id=group, valid_sm=1, valid_et=1)
            for group, frequency, dates in (("a", 0, 1), ("b", 0, 5), ("rare", 1, 8))
            for _ in range(dates)]
    dataset.sites = [SimpleNamespace(table=pd.DataFrame(rows))]
    dataset.offsets = [0, 2, 16]
    weights = dataset.sampling_weights()
    assert weights.sum() == pytest.approx(1)
    assert weights[2] == pytest.approx(1 / 6)
    assert weights[3:8].sum() == pytest.approx(1 / 6)
    assert weights[8:].sum() == pytest.approx(1 / 6)


def test_station_priority_score_and_equal_group_report(joint):
    _, stats, _ = joint
    tasks = Config.fromfile("configs/hydrostate_joint.py").tasks
    metric = JointHydroStateMetric(statistics(), stats, tasks,
                                  comparison_years=(2017, 2024))

    def row(source, task, error, group, year):
        return dict(source=source, spatial_group_id=group, frequency_bin="0", year=year,
                    **{f"{task}/count": 1, f"{task}/sum_abs": abs(error),
                       f"{task}/sum_sq": error**2, f"{task}/sum_err": error,
                       f"{task}/sum_y": 1, f"{task}/sum_y2": 1,
                       f"{task}/sum_x": 1 + error, f"{task}/sum_x2": (1 + error)**2,
                       f"{task}/sum_xy": 1 + error})

    rows = [row("product", "soil_moisture", 1, "", 2020),
            row("site", "site_sm", 1, "long", 2020),
            row("site", "site_sm", 1, "long", 2020),
            row("site", "site_sm", 3, "short", 2015)]
    values = metric.compute_metrics(rows)
    assert values["site/site_sm_rmse"] == pytest.approx((11 / 3)**0.5)
    assert values["site/group_macro/site_sm_rmse"] == 2
    assert values["site/group_macro/site_sm_groups"] == 2
    assert values["site/years_2017_2024/site_sm_rmse"] == 1
    assert values["joint_score"] == pytest.approx(
        values["product/hydro_score"] / 3 + 2 * values["site/hydro_score"] / 3)


def test_gpu_smoke_selection_pairs_both_sources(joint):
    from hydrostate.data.site.training_smoke import representative_indices

    dataset, _, _ = joint
    indices = representative_indices(dataset)
    assert all(index < len(dataset.product) for index in indices[::2])
    assert all(index >= len(dataset.product) for index in indices[1::2])
    assert {dataset.product.table.iloc[i].sampling_group for i in indices[::2]} == set(
        dataset.product.table.sampling_group
    )
