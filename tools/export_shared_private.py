#!/usr/bin/env python
"""Export a version-checked adapter bundle from a normal MMEngine checkpoint."""

import argparse

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_target_statistics
from hydrostate.models.shared_private_model import SharedPrivateHydroStateModel
from hydrostate.models.shared_private_transfer import save_adapter_bundle
from hydrostate.registry import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Prefer the frozen, expanded run configuration")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    configure_target_statistics(cfg, training=False)
    cfg.model.adapter_init = None  # The full checkpoint already contains imported branches.
    init_default_scope(cfg.get("default_scope", "hydrostate"))
    model = MODELS.build(cfg.model)
    if not isinstance(model, SharedPrivateHydroStateModel):
        raise TypeError("This exporter only accepts SharedPrivateHydroStateModel")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    save_adapter_bundle(model, args.output, args.tasks)
    print(f"Exported {args.tasks or list(model.tasks)} to {args.output}")


if __name__ == "__main__":
    main()
