"""Check physical support, stable selection and the real validation lifecycle."""

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader

from hydrostate.datasets.collate import hydro_collate
from hydrostate.engine import HydroVisualizationHook
from hydrostate.engine.visualization_hook import _cpu_sample, comparison_maps
from hydrostate.models.data_preprocessor import HydroDataPreprocessor
from hydrostate.models.hydrostate_model import HydroStateModel
from hydrostate.normalization import LEGACY_TASK_NAMES as TASKS


def _prediction(sample_id="sample", missing_sm=False):
    spatial = torch.tensor([[[[0.1, 0.3], [0.5, 0.9]]]])
    model = SimpleNamespace(
        _forward=lambda inputs: {task: spatial * (10 if task == "et" else 1) for task in TASKS}
    )
    targets = dict(
        water=torch.full_like(spatial, 0.2),
        water_valid=torch.ones_like(spatial, dtype=torch.bool),
        soil_moisture=torch.tensor([[[[0.25]]]]),
        soil_moisture_valid=torch.tensor([[[[not missing_sm]]]]),
        prediction_valid=torch.tensor([[[[True, True], [True, False]]]]),
        et_native_weights=torch.tensor(
            [[[[0.25, 0.75], [0.0, 0.0]], [[0.0, 0.0], [0.5, 0.5]], [[0.0, 0.0], [0.0, 0.0]]]]
        ),
        et_native_value=torch.tensor([[4.0, float("nan"), float("nan")]]),
        et_native_valid=torch.tensor([[True, True, False]]),
        metainfo=dict(sample_id=[sample_id], target_time=["2026-01-01"]),
    )
    return HydroStateModel.predict(model, {}, targets)[0]


def _runner(tmp_path, rank=0, epoch=1):
    return SimpleNamespace(
        work_dir=str(tmp_path),
        rank=rank,
        epoch=epoch,
        iter=4,
        train_loop=None,
        logger=logging.getLogger("visualization-test"),
    )


def test_comparison_uses_actual_supervision_and_input_masks():
    sample = _cpu_sample(_prediction())
    fine, pooled, target = comparison_maps(sample, "water")
    np.testing.assert_allclose(fine, [[0.1, 0.3], [0.5, np.nan]])
    np.testing.assert_allclose(pooled, fine)
    np.testing.assert_allclose(target, [[0.2, 0.2], [0.2, np.nan]])

    fine, pooled, target = comparison_maps(sample, "soil_moisture")
    assert fine.shape == (2, 2)
    assert pooled.shape == target.shape == (1, 1)
    np.testing.assert_allclose(pooled, [[0.3]])  # Only three observed pixels.
    np.testing.assert_allclose(target, [[0.25]])

    fine, pooled, target = comparison_maps(sample, "et")
    np.testing.assert_allclose(fine, [[1.0, 3.0], [5.0, np.nan]])
    np.testing.assert_allclose(pooled, [[2.5, 2.5], [np.nan, np.nan]])
    np.testing.assert_allclose(target, [[4.0, 4.0], [np.nan, np.nan]])
    # The partially observed second native cell and padded NaNs are excluded.
    assert np.isnan(target[1]).all()


def test_missing_labels_and_bfloat16_are_displayable():
    sample = _prediction(missing_sm=True)
    for task in TASKS:
        sample[task] = sample[task].to(torch.bfloat16)
    sample = _cpu_sample(sample)
    fine, pooled, target = comparison_maps(sample, "soil_moisture")
    assert np.isfinite(fine).any()
    assert np.isnan(pooled).all() and np.isnan(target).all()
    sample["valid_et"][:] = False
    _, pooled, target = comparison_maps(sample, "et")
    assert np.isnan(pooled).all() and np.isnan(target).all()


def test_fixed_samples_across_epochs_and_resume(tmp_path, monkeypatch):
    rendered = []
    monkeypatch.setattr(
        HydroVisualizationHook,
        "_draw",
        lambda self, sample, path: rendered.append(sample["metainfo"]["sample_id"]),
    )
    samples = [
        _prediction("a", missing_sm=True),
        _prediction("d"),
        _prediction("c"),
        _prediction("b"),
    ]
    hook = HydroVisualizationHook(num_samples=2)
    for epoch, batch in ((1, samples), (2, samples[::-1])):
        runner = _runner(tmp_path, epoch=epoch)
        hook.before_val_epoch(runner)
        for sample in batch:
            hook.after_val_iter(runner, 0, outputs=[sample])
            assert len(hook._candidates) <= 2
        hook.after_val_epoch(runner)
        report = json.loads(
            (tmp_path / f"visualizations/epoch_{epoch:03d}/samples.json").read_text()
        )
        assert [record["sample_id"] for record in report] == ["b", "c"]
    resumed = HydroVisualizationHook(num_samples=2)
    runner = _runner(tmp_path, epoch=3)
    resumed.before_val_epoch(runner)
    resumed.after_val_iter(runner, 0, outputs=samples[1:] + samples)
    resumed.after_val_epoch(runner)
    assert rendered == ["b", "c"] * 3


@pytest.mark.parametrize(
    "kwargs",
    [dict(num_samples=-1), dict(num_samples=1.5), dict(interval=0), dict(et_vmax=float("nan"))],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        HydroVisualizationHook(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(num_samples=0), dict(interval=2)])
def test_disabled_or_skipped_hook_does_no_work(tmp_path, kwargs):
    hook = HydroVisualizationHook(**kwargs)
    runner = _runner(tmp_path)
    hook.before_val_epoch(runner)
    hook.after_val_iter(runner, 0, outputs=None)
    hook.after_val_epoch(runner)
    assert not (tmp_path / "visualizations").exists()


def test_query_times_are_moved_to_cpu_and_saved_per_variable(tmp_path, monkeypatch):
    sample = _prediction()
    sample["target_time"] = {task: torch.tensor([1600000000, 1600000000]) for task in TASKS}
    sample["target_time"]["et"][1] += 8 * 86400
    cpu = _cpu_sample(sample)
    assert isinstance(cpu["target_time"]["et"], np.ndarray)
    hook = HydroVisualizationHook(num_samples=1)
    runner = _runner(tmp_path)
    monkeypatch.setattr(hook, "_draw", lambda sample, path: None)
    hook.before_val_epoch(runner)
    hook.after_val_iter(runner, 0, outputs=[sample])
    hook.after_val_epoch(runner)
    report = json.loads((tmp_path / "visualizations/epoch_001/samples.json").read_text())
    assert report[0]["target_time"]["et"] == [1600000000, 1600000000 + 8 * 86400]


def test_l40s_config_and_smoke_preserve_hook(monkeypatch):
    monkeypatch.setenv("HYDROSTATE_VIS_NUM_SAMPLES", "7")
    root = Path(__file__).resolve().parents[1]
    for filename in ("hydrostate_l40s.py", "hydrostate_l40s_smoke.py"):
        cfg = Config.fromfile(root / "configs/experiments/legacy" / filename)
        assert cfg.custom_hooks[0].type == "HydroVisualizationHook"
        assert cfg.custom_hooks[0].num_samples == 7
        cfg.merge_from_dict({"custom_hooks.0.num_samples": 2})
        assert cfg.custom_hooks[0].num_samples == 2
    assert cfg.custom_hooks[1].type == "NLLVerificationHook"


class _TinyModel(HydroStateModel):
    """Exercise the real preprocessor/predict path with a cheap trainable forward."""

    def __init__(self):
        BaseModel.__init__(self, data_preprocessor=HydroDataPreprocessor())
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def _forward(self, inputs):
        self.forward_count += 1
        pixels = torch.nn.functional.avg_pool2d(inputs["s1"][:, 0, :1], 3)
        return {task: pixels * self.scale for task in TASKS}

    def loss(self, inputs, data_samples):
        return dict(loss=self._forward(inputs)["water"].square().mean())


def _training_sample(index):
    return dict(
        inputs={
            **{key: torch.ones(1, 1, 6, 6) for key in ("s1", "s2", "landsat")},
            **{
                f"{key}_valid": torch.ones(1, 1, 6, 6, dtype=torch.bool)
                for key in ("s1", "s2", "landsat")
            },
            "sensor_valid": torch.ones(1, 3, dtype=torch.bool),
        },
        data_samples=dict(
            water=torch.full((1, 6, 6), 0.2),
            water_valid=torch.ones(1, 6, 6, dtype=torch.bool),
            soil_moisture=torch.full((1, 1, 1), 0.3),
            soil_moisture_valid=torch.ones(1, 1, 1, dtype=torch.bool),
            et_native_weights=torch.full((1, 2, 2), 0.25),
            et_native_value=torch.tensor([2.0]),
            et_native_valid=torch.tensor([True]),
            metainfo=dict(sample_id=f"sample_{index}", target_time="2026-01-01"),
        ),
    )


@pytest.mark.parametrize("by_epoch", [True, False])
def test_real_runner_writes_nine_panel_pngs(tmp_path, target_statistics, by_epoch):
    samples = [_training_sample(index) for index in range(3)]
    model = _TinyModel()
    model.forward_count = 0
    runner = Runner(
        model=model,
        work_dir=str(tmp_path),
        train_dataloader=DataLoader(copy.deepcopy(samples), batch_size=2, collate_fn=hydro_collate),
        val_dataloader=DataLoader(samples, batch_size=2, collate_fn=hydro_collate),
        train_cfg=(
            dict(by_epoch=True, max_epochs=2, val_interval=1)
            if by_epoch
            else dict(by_epoch=False, max_iters=2, val_interval=2)
        ),
        val_cfg=dict(),
        val_evaluator=dict(type="HydroStateMetric", target_statistics=target_statistics),
        optim_wrapper=dict(optimizer=dict(type="SGD", lr=0.01)),
        custom_hooks=[HydroVisualizationHook(num_samples=2 if by_epoch else 4)],
        default_hooks=dict(checkpoint=None, logger=dict(type="LoggerHook", interval=1)),
        default_scope="hydrostate",
        log_processor=dict(by_epoch=by_epoch),
        visualizer=dict(type="Visualizer", vis_backends=[]),
        env_cfg=dict(dist_cfg=dict(backend="gloo")),
    )
    runner.train()
    assert model.forward_count == (8 if by_epoch else 4)  # No extra visualization inference.
    from matplotlib.image import imread

    labels = ["epoch_001", "epoch_002"] if by_epoch else ["iter_000002"]
    for label in labels:
        directory = tmp_path / "visualizations" / label
        paths = sorted(directory.glob("*.png"))
        assert len(paths) == (2 if by_epoch else 3)
        assert imread(paths[0]).shape[:2] == (1540, 1680)
        report = json.loads((directory / "samples.json").read_text())
        assert [record["sample_id"] for record in report] == [
            f"sample_{index}" for index in range(2 if by_epoch else 3)
        ]


def _distributed_visualization(rank, init_file, directory):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        hook = HydroVisualizationHook(num_samples=3)
        runner = _runner(directory, rank=rank)
        hook._draw = lambda sample, path: Path(path).write_text(f"rank={rank}")
        hook.before_val_epoch(runner)
        samples = (
            [_prediction("c"), _prediction("a", missing_sm=True)]
            if rank == 0
            else [_prediction("b"), _prediction("c")]
        )
        hook.after_val_iter(runner, 0, outputs=samples)
        hook.after_val_epoch(runner)
    finally:
        dist.destroy_process_group()


def test_distributed_selection_deduplicates_and_only_rank_zero_writes(tmp_path):
    mp.spawn(
        _distributed_visualization,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    directory = tmp_path / "visualizations/epoch_001"
    records = json.loads((directory / "samples.json").read_text())
    assert [record["sample_id"] for record in records] == ["b", "c", "a"]
    assert all(path.read_text() == "rank=0" for path in directory.glob("*.png"))
