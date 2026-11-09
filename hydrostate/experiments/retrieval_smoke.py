"""Bounded real-sample forward/backward checks; never a training release."""

import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed

from hydrostate.cli import configure_data_contract, configure_pretrained_encoder
from hydrostate.datasets.collate import hydro_collate
from hydrostate.registry import DATASETS, MODELS


def smoke(config, data_root, pretrained, output, *, samples=4, batch_size=1, device="cuda"):
    """Use a small training subset and existing statistics without publishing weights.

    Subsets deliberately do not pass the full training-release fingerprint check;
    this validates numerical execution only, never creates new target statistics,
    and cannot be used to claim generalization or a production batch-size limit.
    """
    if min(samples, batch_size) < 1:
        raise ValueError("Sample count and batch size must be positive")
    root = Path(data_root)
    cfg = Config.fromfile(config)
    configure_data_contract(cfg)
    configure_pretrained_encoder(cfg, pretrained)
    init_default_scope("hydrostate")
    set_random_seed(cfg.randomness.seed, deterministic=cfg.randomness.get("deterministic", False))
    cfg.model.target_statistics = json.loads((root / "target_stats.json").read_text())
    dataset_cfg = dict(cfg.train_dataloader.dataset)
    dataset_cfg.update(data_root=str(root), manifest=str(root / "samples.parquet"), training=False)
    dataset = DATASETS.build(dataset_cfg)
    samples = min(samples, len(dataset))
    if samples == 0:
        raise ValueError("Smoke subset has no training samples")
    model = MODELS.build(cfg.model).to(device).train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), **{
            key: value for key, value in cfg.optim_wrapper.optimizer.items() if key != "type"})
    cuda = torch.device(device).type == "cuda"
    context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cuda else nullcontext
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    report = dict(
        config=str(config), pretrained=str(pretrained), data_root=str(root), device=str(device),
        gpu=torch.cuda.get_device_name() if cuda else None,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        batch_size=batch_size, steps=[],
        scope="Real-data numerical smoke only; not a full run or batch-size calibration",
    )
    for start in range(0, samples, batch_size):
        records = [dataset[i] for i in range(start, min(start + batch_size, samples))]
        data = model.data_preprocessor(hydro_collate(records), training=True)
        if cuda:
            torch.cuda.synchronize()
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with context():
            losses = model(**data, mode="loss")
            loss = sum(v for name, v in losses.items() if name.startswith("loss_"))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite real-sample loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        row = dict(start=start, samples=len(records), seconds=time.monotonic() - began,
                   grad_norm=float(norm), losses={name: float(v.detach())
                                                 for name, v in losses.items()})
        report["steps"].append(row)
        print(json.dumps(row), flush=True)
        del loss, losses
    model.eval()
    with torch.inference_mode(), context():
        predictions = model(data["inputs"], mode="predict")
    report["prediction_shapes"] = {
        name: list(predictions[0][name].shape) for name in (*model.tasks, "et_total")}
    for prediction in predictions:
        for task in model.tasks:
            mask = prediction.get(f"{task}_prediction_valid", prediction["prediction_valid"])
            if not torch.isfinite(prediction[task][mask]).all():
                raise RuntimeError(f"Nonfinite valid prediction: {task}")
    if hasattr(model, "router"):
        report["router_gamma"] = {name: float(value.detach())
                                  for name, value in model.router.gamma.items()}
        report["router_enabled"] = model.router.enabled
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30 if cuda else None
    report["passed"] = True
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    del optimizer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return report
