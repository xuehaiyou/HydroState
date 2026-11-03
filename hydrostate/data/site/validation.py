"""A real station Zarr -> model -> backward check; never an accuracy experiment."""

import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.registry import MODELS

from .statistics import station_statistics


def validate_training_path(manifest, config, product_statistics, output, *, batch_size=2):
    """Use actual training-only site scales; product scales are explicit, frozen input."""
    torch.set_num_threads(2)
    torch.manual_seed(43)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = SiteDataset(manifest, "train")
    stats = station_statistics(manifest)
    atomic_json(output / "site_target_stats.json", stats)
    cfg = Config.fromfile(str(config))
    init_default_scope("hydrostate")
    cfg.model.target_statistics = str(product_statistics)
    cfg.model.site_supervision = dict(statistics=stats)
    model = MODELS.build(cfg.model).train()
    samples = hydro_collate([dataset[i] for i in range(min(batch_size, len(dataset)))])
    data = model.data_preprocessor(samples, training=True)
    losses = model.loss(**data)
    objective = sum(value for name, value in losses.items() if name.startswith("loss_"))
    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("Nonfinite loss on real station input")
    objective.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("Invalid/missing model gradients")
    if not any(gradient.abs().sum() > 0 for gradient in gradients):
        raise ValueError("Station loss did not train the model")
    with torch.no_grad():
        predictions = model.predict(**data)
    result = dict(
        status="passed",
        scope="CPU_forward_backward_only_not_accuracy_or_training",
        architecture=cfg.model.type,
        batch=len(predictions),
        parameters=sum(p.numel() for p in model.parameters()),
        input_grid=[120, 120],
        output_grid=[40, 40],
        station_statistics=stats,
        losses={k: float(v.detach()) for k, v in losses.items()},
        finite_gradients=True,
        nonzero_gradients=True,
        source_manifest=str(manifest),
        product_statistics=str(product_statistics),
    )
    atomic_json(output / "training_path.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                batch=result["batch"],
                parameters=result["parameters"],
                loss=float(objective.detach()),
            )
        ),
        flush=True,
    )
    return result
