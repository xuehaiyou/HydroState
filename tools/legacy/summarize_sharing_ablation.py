#!/usr/bin/env python
"""Aggregate the three sharing modes and paired seed/spatial-block uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrostate.data.sampling_manifest import geographic_block
from tools.legacy.submit_shared_private import save_record
from tools.legacy.submit_sharing_ablation import TASKS, sha256

GROUPS = ("shared", "shared_private", "independent")


def paired_summary(cube, scales, draws=2000, seed=4817):
    """Cube axes: group, training seed, spatial block, task, (squared error, count)."""
    if (cube.shape[0] != 3 or cube.shape[-2:] != (3, 2) or cube.shape[1] < 2
            or cube.shape[2] < 2 or draws < 1 or not np.isfinite(cube).all()):
        raise ValueError("Require three groups/tasks, repeated seeds, and spatial blocks")
    if not np.array_equal(cube[..., 1], np.broadcast_to(cube[2, 0, :, :, 1], cube[..., 1].shape)):
        raise ValueError("Evaluation supports differ across groups or seeds")
    totals = cube.sum(axis=2)
    if np.any(totals[..., 1] <= 0):
        raise ValueError("Every task needs test observations")
    rmse = np.sqrt(totals[..., 0] / totals[..., 1])
    score = (rmse / np.asarray(scales)).mean(axis=-1)
    pairs = ((0, 2), (1, 2), (1, 0))
    samples = {pair: [] for pair in pairs}
    rng = np.random.default_rng(seed)
    # Resample identical blocks and seeds in every group to preserve pairing.
    for _ in range(draws * 10):
        indices = rng.integers(cube.shape[1], size=cube.shape[1])
        weights = np.bincount(rng.integers(cube.shape[2], size=cube.shape[2]),
                              minlength=cube.shape[2])
        selected = np.einsum("gsbtk,b->gstk", cube[:, indices], weights)
        if np.any(selected[..., 1] <= 0):
            continue
        values = np.sqrt(selected[..., 0] / selected[..., 1])
        combined = np.concatenate([values, (values / scales).mean(axis=-1)[..., None]], axis=-1)
        for a, b in pairs:
            samples[(a, b)].append((combined[a] - combined[b]).mean(axis=0))
        if len(samples[pairs[0]]) == draws:
            break
    if len(samples[pairs[0]]) != draws:
        raise ValueError("Too few valid spatial bootstrap draws")
    return dict(
        groups={group: dict(rmse_mean=rmse[g].mean(axis=0).tolist(),
                           rmse_seed_std=rmse[g].std(axis=0, ddof=1).tolist(),
                           rmse_by_seed=rmse[g].tolist(),
                           normalized_rmse_mean=float(score[g].mean()),
                           gain_vs_independent_percent=(
                               100 * (rmse[2] - rmse[g]) / rmse[2]).mean(axis=0).tolist())
                for g, group in enumerate(GROUPS)},
        paired_differences={f"{GROUPS[a]}_minus_{GROUPS[b]}": dict(
            labels=[*TASKS, "mean_normalized_rmse"], lower_is_better=True,
            ci95=np.quantile(samples[(a, b)], [.025, .975], axis=0).T.tolist())
                            for a, b in pairs},
        bootstrap=dict(draws=draws, seed=seed, blocks=cube.shape[2], training_seeds=cube.shape[1],
                       method="paired training seeds and 5-degree spatial blocks",
                       multiplicity_adjusted=False),
    )


def summarize(receipt, draws=2000):
    record = json.loads(receipt.read_text())
    if sha256(record["manifest"]) != record["manifest_sha256"]:
        raise ValueError("Manifest changed since submission")
    if sha256(record["statistics"]) != record["statistics_sha256"]:
        raise ValueError("Training statistics changed since submission")
    table = pd.read_parquet(record["manifest"])
    table = table.loc[table.split == "test"].copy()
    locations = {row["sample_id"]: geographic_block(row) for row in table.to_dict("records")}
    blocks = {block: i for i, block in enumerate(sorted(set(locations.values())))}
    expected_ids = set(locations)
    seeds = record["seeds"]
    scales = json.loads(Path(record["statistics"]).read_text())["tasks"]
    scales = np.array([scales[t]["std"] for t in TASKS])
    result = dict(commit=record["commit"], seeds=seeds, tasks=TASKS,
                  training_fingerprint=record["training_fingerprint"], evaluations={}, runs=[])
    for kind in ("final", "best"):
        cube = np.zeros((3, len(seeds), len(blocks), 3, 2), dtype=np.float64)
        seen = set()
        for entry in record["experiments"]:
            metrics_file = Path(entry["work_dir"]) / "evaluation/metrics.json"
            metrics = json.loads(metrics_file.read_text())
            if (metrics["seed"] != entry["seed"]
                    or metrics["training_fingerprint"] != record["training_fingerprint"]
                    or metrics["selected_tasks"] != entry["selected_tasks"]
                    or metrics["config_sha256"] != entry["config_sha256"]):
                raise ValueError(f"Evaluation provenance mismatch: {entry['stage']}")
            report = metrics["results"][kind]
            if kind == "final":
                result["runs"].append(dict(stage=entry["stage"],
                                           training_wall_seconds=metrics["training_wall_seconds"],
                                           parameters=report["parameters_by_owner"],
                                           metrics=report["metrics"]))
            rows = [json.loads(line)
                    for line in Path(report["sample_metrics"]).read_text().splitlines()]
            if len(rows) != len(expected_ids) or {r["sample_id"] for r in rows} != expected_ids:
                raise ValueError("Test sample identities differ across runs")
            g, s = GROUPS.index(entry["group"]), seeds.index(entry["seed"])
            for task in entry["selected_tasks"]:
                key = (g, s, task)
                if key in seen:
                    raise ValueError("Duplicate task result")
                seen.add(key)
                t = TASKS.index(task)
                for row in rows:
                    b = blocks[locations[row["sample_id"]]]
                    cube[g, s, b, t] += [row.get(f"{task}/sum_sq", 0),
                                         row.get(f"{task}/count", 0)]
            for task in entry["selected_tasks"]:
                t = TASKS.index(task)
                summed = cube[g, s, :, t].sum(axis=0)
                if not np.isclose(np.sqrt(summed[0] / summed[1]),
                                  report["metrics"][f"{task}_rmse"]):
                    raise ValueError("Saved sample errors disagree with reported RMSE")
        if len(seen) != 3 * len(seeds) * 3:
            raise ValueError("Missing group/seed/task results")
        result["evaluations"][kind] = paired_summary(cube, scales, draws=draws)
    destination = receipt.parent / "comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    save_record(destination, result)
    print(f"Comparison: {destination}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    summarize(args.receipt, args.bootstrap_draws)


if __name__ == "__main__":
    main()
