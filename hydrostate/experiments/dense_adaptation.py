"""Run one six-epoch dense adaptation comparison against the published four-stage control."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.experiments.foundation_search import probe
from hydrostate.experiments.manual_cover_launch import finish_test

MULTIFM = ("multifm_rutal", "multifm_rutal_equal")
ARMS = ("nexus_d2moe", "faar", "rutal", "bdta", *MULTIFM)


def configure(reference, directory, arm):
    cfg = Config.fromfile(reference)
    if not (
        cfg.model.encoder.method == "d2moe"
        and cfg.model.type == "FoundationStageModelV2"
        and cfg.model.encoder.out_indices == [2, 5, 8, 11]
        and not cfg.model.router.enabled
        and cfg.randomness.seed == 3407
        and cfg.train_cfg.max_epochs == 6
        and cfg.train_cfg.val_interval == 1
        and not cfg.resume
        and cfg.load_from is None
        and cfg.model.manual_cover_weight == 2
        and cfg.model.site_supervision
        and cfg.train_dataloader.dataset.type == "ManualJointDataset"
        and cfg.default_hooks.checkpoint.save_best == "manual_joint_score"
    ):
        raise ValueError("Expected the matched six-epoch D2MoE v2 experiment")
    variant = Config.fromfile(f"configs/hydrostate_joint_{arm}.py")
    cfg.model.type = variant.model.type
    if arm == "nexus_d2moe":
        cfg.model.alignment_weight = variant.model.alignment_weight
    else:
        cfg.model.encoder.method = variant.model.encoder.method
        cfg.model.decoder.update(
            type=variant.model.decoder.type,
            channels=variant.model.decoder.channels,
            num_levels=variant.model.decoder.num_levels,
        )
    if arm == "bdta":
        cfg.model.encoder.gradient_checkpointing = False
    if arm in MULTIFM:
        cfg.model.anysat = variant.model.anysat
        cfg.model.decoder.learn_source_weights = variant.model.decoder.learn_source_weights
        if not (cfg.val_dataloader.batch_size == cfg.test_dataloader.batch_size == 8):
            raise ValueError("Expected the baseline validation and test batch of eight")
    if arm == "faar":
        cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(type="FAARRankHook")]
    cfg.work_dir = str(directory / "work")
    return cfg


def run(reference, directory, arm):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = configure(reference, directory, arm)
    cfg.dump(str(directory / "config.py"))
    probe(directory)
    accepted = json.loads((directory / "accepted.json").read_text())
    bench = json.loads(Path(accepted["report"]).read_text())
    if (bench["train_samples"], bench["validation_samples"]) != (10912, 3681):
        raise ValueError("Mismatch with the original quick-test subset")
    batch = accepted["batch"]
    cfg.train_dataloader.batch_size = batch
    cfg.optim_wrapper.accumulative_counts = 10 // batch
    cfg.param_scheduler[0].end = 100 * (10 // batch)
    path = directory / "train_config.py"
    cfg.dump(str(path))
    atomic_json(
        directory / "run.json",
        dict(
            arm=arm,
            reference=str(reference),
            gpu=bench["gpu"],
            microbatch=batch,
            effective_batch=10,
            epochs=6,
            validation_batch=cfg.val_dataloader.batch_size,
            selection=cfg.default_hooks.checkpoint.save_best,
            config=str(path),
        ),
    )
    subprocess.run([sys.executable, "tools/train.py", str(path)], check=True)
    finish_test(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("reference", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.reference, args.directory, args.arm)
