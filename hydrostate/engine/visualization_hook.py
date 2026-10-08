"""Fixed validation samples at both the output and observational resolutions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from mmengine.dist import all_gather_object
from mmengine.hooks import Hook
from mmengine.runner import IterBasedTrainLoop

from hydrostate.normalization import TASKS
from hydrostate.registry import HOOKS


def _masked(values, valid):
    return np.where(np.asarray(valid, dtype=bool), values, np.nan)


def _native_map(values, weights, valid):
    """Project valid native-cell scalars onto their footprints for display only.

    At cell boundaries, blend using the stored normalized area weights. This
    projection does not create additional observations or change supervision.
    """
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    weights = weights[valid]
    values = values.reshape(-1)[valid]
    coverage = weights.sum(axis=0)
    result = np.full(coverage.shape, np.nan, dtype=np.float32)
    np.divide(
        (weights * values[:, None, None]).sum(axis=0), coverage, out=result, where=coverage > 0
    )
    return result


def comparison_maps(sample, task):
    """Return 30 m prediction, supervision-scale prediction and ground truth.

    The scalar SM panels deliberately stay 1 x 1. ET uses the same footprint
    projection for both the already-pooled predictions and native targets.
    """
    prediction = _masked(sample[task], sample["prediction_valid"])[0]
    supervised, target, valid = (
        sample[f"{prefix}_{task}"] for prefix in ("supervised", "target", "valid")
    )
    if task == "et":
        weights = sample["et_native_weights"]
        return (
            prediction,
            _native_map(supervised, weights, valid),
            _native_map(target, weights, valid),
        )
    return prediction, _masked(supervised, valid)[0], _masked(target, valid)[0]


def _cpu_sample(sample):
    result = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            value = value.detach()
            if value.is_floating_point():
                value = value.float()  # NumPy does not support torch.bfloat16.
            result[key] = value.cpu().numpy()
        else:
            result[key] = value
    return result


@HOOKS.register_module()
class HydroVisualizationHook(Hook):
    """Save a nine-panel PNG per fixed sample after each validation epoch.

    Select deterministically by decreasing number of supervised tasks, then
    sample_id. With an immutable validation split, the same samples are chosen
    across epochs, restarts and DDP world sizes. Only bounded CPU candidates are
    retained; inference is reused from validation and only rank zero writes.
    """

    def __init__(self, num_samples=4, interval=1, out_dir="visualizations", et_vmax=10.0):
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer (0 disables plotting)")
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be a positive integer")
        if not np.isfinite(et_vmax) or et_vmax <= 0:
            raise ValueError("et_vmax must be finite and positive")
        self.num_samples = num_samples
        self.interval = interval
        self.out_dir = out_dir
        self.et_vmax = et_vmax
        self._active = False
        self._candidates = {}

    def before_val_epoch(self, runner):
        self._candidates.clear()
        self._active = False
        if self.num_samples == 0:
            return
        by_iter = isinstance(runner.train_loop, IterBasedTrainLoop)
        step = runner.iter if by_iter else runner.epoch
        self._label = f"iter_{step:06d}" if by_iter else f"epoch_{step:03d}"
        self._active = step % self.interval == 0

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._active:
            return
        for sample in outputs:
            sample_id = str(sample["metainfo"]["sample_id"])
            # Validity depends on labels/input coverage, never model quality.
            count = sum(bool(sample[f"valid_{task}"].any()) for task in TASKS)
            key = (-count, sample_id)
            if sample_id in self._candidates:
                continue  # Distributed samplers can pad with duplicate samples.
            if len(self._candidates) >= self.num_samples:
                worst = max(self._candidates, key=lambda name: self._candidates[name][0])
                if key >= self._candidates[worst][0]:
                    continue
                del self._candidates[worst]
            self._candidates[sample_id] = (key, _cpu_sample(sample))

    def after_val_epoch(self, runner, metrics=None):
        if not self._active:
            return
        # All ranks must participate. all_gather_object also supports NCCL.
        gathered = all_gather_object(list(self._candidates.values()))
        self._candidates.clear()
        if runner.rank != 0:
            return
        unique = {}
        for candidates in gathered:
            for key, sample in candidates:
                unique.setdefault(key[1], (key, sample))
        selected = sorted(unique.values(), key=lambda item: item[0])[: self.num_samples]
        directory = Path(runner.work_dir) / self.out_dir / self._label
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index, (key, sample) in enumerate(selected):
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", key[1])[:100]
            filename = f"{index + 1:02d}_{safe_id}.png"
            self._draw(sample, directory / filename)
            records.append(
                dict(
                    sample_id=key[1],
                    file=filename,
                    metainfo=sample["metainfo"],
                    valid_counts={task: int(sample[f"valid_{task}"].sum()) for task in TASKS},
                )
            )
        (directory / "samples.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if len(selected) < self.num_samples:
            runner.logger.warning(
                "Requested %d visualization samples; only %d are available",
                self.num_samples,
                len(selected),
            )
        runner.logger.info("Saved %d WF/SM/ET comparisons to %s", len(selected), directory)

    def _draw(self, sample, path):
        # Agg works on headless SLURM nodes without changing global pyplot state.
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(12, 11), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 3)
        meta = sample["metainfo"]
        figure.suptitle(
            f"{self._label} | {meta['sample_id']}\n{meta.get('target_time', '')} | validation",
            fontsize=13,
        )
        titles = (
            (
                "WF prediction - 30 m",
                "WF prediction - supervision (30 m)",
                "WF ground truth - 30 m",
            ),
            (
                "SM prediction - 30 m",
                "SM prediction - valid-window mean",
                "SM ground truth - SMAP center scalar",
            ),
            (
                "ET prediction - 30 m",
                "ET prediction - native-cell area mean",
                "ET ground truth - native cells",
            ),
        )
        for row, (task, unit, palette, vmax) in enumerate(
            zip(
                TASKS,
                ("fraction", "m³/m³", "mm/day"),
                ("Blues", "YlGnBu", "YlOrRd"),
                (1.0, 1.0, self.et_vmax),
            )
        ):
            cmap = colormaps[palette].with_extremes(bad="#d9d9d9")
            for column, values in enumerate(comparison_maps(sample, task)):
                axis = axes[row, column]
                artist = axis.imshow(
                    np.ma.masked_invalid(values),
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                )
                axis.set_title(titles[row][column], fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
                if not np.isfinite(values).any():
                    axis.text(
                        0.5,
                        0.5,
                        "No valid data",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                elif task == "soil_moisture" and column > 0:
                    axis.text(
                        0.5,
                        0.5,
                        f"{float(values.item()):.4f}\n{unit}\n(single scalar)",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                    )
            figure.colorbar(artist, ax=list(axes[row]), label=unit, shrink=0.8, extend="max")
        figure.supxlabel(
            "Gray: invalid/unobserved. Fixed color limits across epochs.\n"
            "SM scalar panels are not spatial maps. ET native panels use footprint projection; "
            "boundary pixels are blended.",
            fontsize=9,
        )
        try:
            figure.savefig(path, dpi=140)
        finally:
            figure.clear()
