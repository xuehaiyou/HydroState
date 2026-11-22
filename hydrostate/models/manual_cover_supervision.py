"""One source-independent, per-window human fraction objective."""

import torch

COVER = ("water", "snow_ice", "land")


def manual_cover_loss(outputs, samples, weight=2.0):
    errors, present = [], []
    support = samples.get("prediction_valid", torch.ones_like(outputs["water"], dtype=torch.bool))
    for name in COVER:
        target = samples["manual_" + name]
        valid = samples["manual_" + name + "_valid"].bool() & support
        count = valid.sum(dim=(1, 2, 3))
        # Mask before reduction, so missing labels contribute neither zeros nor NaNs.
        difference = torch.where(valid, outputs[name] - target, 0.0)
        errors.append(difference.square().sum(dim=(1, 2, 3)) / count.clamp_min(1))
        present.append(count > 0)
    valid_tasks = torch.stack(present, dim=1)
    per_window = torch.stack(errors, dim=1).sum(1) / valid_tasks.sum(1).clamp_min(1)
    valid_windows = valid_tasks.any(1)
    return weight * per_window.sum() / valid_windows.sum().clamp_min(1)
