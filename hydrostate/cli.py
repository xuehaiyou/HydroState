"""MMEngine command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from hydrostate.normalization import load_target_statistics, training_fingerprint
from hydrostate.tasks import task_config


def configure_data_contract(cfg: Config) -> None:
    """Fan out one optional task/pack configuration without changing legacy configs."""
    tasks = cfg.get("tasks", cfg.model.get("tasks"))
    if tasks is not None:
        tasks = task_config(tasks)
        if cfg.model.get("tasks") is not None and task_config(cfg.model.tasks) != tasks:
            raise ValueError("Top-level tasks disagree with model tasks")
        cfg.model.tasks = tasks
        for hook in cfg.get("custom_hooks", []):
            if str(hook.get("type", "")).endswith("HydroVisualizationHook"):
                hook["tasks"] = tasks
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = cfg.get(key)
        if loader is None:
            continue
        dataset = loader.dataset
        if tasks is not None:
            if dataset.get("tasks") is not None and task_config(dataset.tasks) != tasks:
                raise ValueError(f"{key} tasks disagree with model tasks")
            dataset.tasks = tasks
        for name in ("input_packs", "label_packs", "target_time_queries"):
            if name in cfg:
                if name in dataset and dataset[name] != cfg[name]:
                    raise ValueError(f"{key} {name} disagree with the shared configuration")
                dataset[name] = cfg[name]
        scale = cfg.model.get("output_scale", 3)
        if dataset.get("output_scale", scale) != scale:
            raise ValueError("Dataset and model output_scale disagree")
        if scale != 3:
            dataset.output_scale = scale


def configure_target_statistics(cfg: Config, training: bool) -> None:
    """Freeze one training-statistics version into the model and both evaluators."""
    configure_data_contract(cfg)
    tasks = cfg.model.get("tasks")
    source = cfg.model.target_statistics
    if source == "__manifest__":
        manifest = cfg.train_dataloader.dataset.manifest
        source = str(Path(manifest).with_name("target_stats.json"))
    statistics = load_target_statistics(source, tasks)
    if training:
        import pandas as pd

        dataset = cfg.train_dataloader.dataset
        if dataset.split != "train":
            raise ValueError("Fine-tuning requires the train split")
        contract = None
        if any(dataset.get(key) for key in ("tasks", "input_packs", "label_packs", "target_time_queries")) or (
            dataset.get("output_scale", 3) != 3
        ):
            from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset

            loaded = HydroStateZarrDataset(**{k: v for k, v in dataset.items() if k != "type"})
            contract = loaded.data_contract
            selected_inputs = {name for pack in loaded.input_packs for name in pack.variables}
            fusion = cfg.model.get("input_fusion") or {}
            if selected_inputs != set(fusion.get("inputs", {})):
                raise ValueError("Selected auxiliary inputs must match model.input_fusion.inputs")
        actual = training_fingerprint(pd.read_parquet(dataset.manifest), contract)
        if actual != statistics["training_fingerprint"]:
            raise ValueError("Training manifest differs from frozen target statistics")
    cfg.model.target_statistics = statistics
    for key in ("val_evaluator", "test_evaluator"):
        evaluators = cfg.get(key)
        if evaluators is None:
            continue
        for evaluator in evaluators if isinstance(evaluators, list) else [evaluators]:
            if evaluator["type"].split(".")[-1] == "HydroStateMetric":
                evaluator["target_statistics"] = statistics
                if tasks is not None:
                    evaluator["tasks"] = tasks


def _parser(description: str, checkpoint: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config")
    if checkpoint:
        parser.add_argument("checkpoint")
    parser.add_argument("--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"])
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    return parser


def _config(args) -> Config:
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    return cfg


def train_main() -> None:
    args = _parser("Train HydroState").parse_args()
    cfg = _config(args)
    configure_target_statistics(cfg, training=True)
    Runner.from_cfg(cfg).train()


def test_main() -> None:
    args = _parser("Test HydroState", checkpoint=True).parse_args()
    cfg = _config(args)
    configure_target_statistics(cfg, training=False)
    cfg.load_from = args.checkpoint
    Runner.from_cfg(cfg).test()


def predict_main() -> None:
    # Prediction uses MMEngine's test loop; configure a prediction manifest and
    # output hook in a derived config when regional export is implemented.
    args = _parser("Predict HydroState", checkpoint=True).parse_args()
    cfg = _config(args)
    configure_target_statistics(cfg, training=False)
    cfg.load_from = args.checkpoint
    Runner.from_cfg(cfg).test()
