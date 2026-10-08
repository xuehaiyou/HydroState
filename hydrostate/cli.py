"""MMEngine command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from hydrostate.normalization import load_target_statistics, training_fingerprint


def configure_target_statistics(cfg: Config, training: bool) -> None:
    """Freeze one training-statistics version into the model and both evaluators."""
    source = cfg.model.target_statistics
    if source == "__manifest__":
        manifest = cfg.train_dataloader.dataset.manifest
        source = str(Path(manifest).with_name("target_stats.json"))
    statistics = load_target_statistics(source)
    if training:
        import pandas as pd

        dataset = cfg.train_dataloader.dataset
        if dataset.split != "train":
            raise ValueError("Fine-tuning requires the train split")
        actual = training_fingerprint(pd.read_parquet(dataset.manifest))
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
