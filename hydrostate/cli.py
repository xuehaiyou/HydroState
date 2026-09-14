"""MMEngine command-line entry points."""

from __future__ import annotations

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


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
    Runner.from_cfg(_config(args)).train()


def test_main() -> None:
    args = _parser("Test HydroState", checkpoint=True).parse_args()
    cfg = _config(args)
    cfg.load_from = args.checkpoint
    Runner.from_cfg(cfg).test()


def predict_main() -> None:
    # Prediction uses MMEngine's test loop; configure a prediction manifest and
    # output hook in a derived config when regional export is implemented.
    args = _parser("Predict HydroState", checkpoint=True).parse_args()
    cfg = _config(args)
    cfg.load_from = args.checkpoint
    Runner.from_cfg(cfg).test()

