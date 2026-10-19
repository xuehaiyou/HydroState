#!/usr/bin/env python
"""Compose independently specialized branches into one inference checkpoint."""

import argparse
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from hydrostate.cli import configure_target_statistics
from hydrostate.models.shared_private_model import SharedPrivateHydroStateModel
from hydrostate.models.shared_private_transfer import load_adapter_bundle
from hydrostate.registry import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Expanded stage-A config with frozen target statistics")
    parser.add_argument("checkpoint", help="Stage-A full checkpoint")
    parser.add_argument("output", type=Path, help="New full inference checkpoint")
    parser.add_argument("--task-bundles", nargs="+", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    configure_target_statistics(cfg, training=False)
    cfg.model.adapter_init = None
    init_default_scope(cfg.get("default_scope", "hydrostate"))
    model = MODELS.build(cfg.model)
    if not isinstance(model, SharedPrivateHydroStateModel):
        raise TypeError("Composition requires SharedPrivateHydroStateModel")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source["state_dict"], strict=True)
    imported = set()
    for path in args.task_bundles:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        names = set(bundle["tasks"])
        if imported & names:
            raise ValueError(f"Conflicting task bundles: {sorted(imported & names)}")
        load_adapter_bundle(model, path, load_shared=False)
        imported.update(names)
    checkpoint = dict(
        state_dict=model.state_dict(),
        meta=dict(config=cfg.pretty_text, composed_tasks=sorted(imported),
                  source_checkpoint=args.checkpoint, task_bundles=args.task_bundles),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Composed {sorted(imported)} into {args.output} (no optimizer/resume state)")


if __name__ == "__main__":
    main()
