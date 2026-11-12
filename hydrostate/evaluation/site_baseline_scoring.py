"""Same-observation product/model comparison; no independent per-method cherry-picking."""

import numpy as np


def agreement(target, prediction):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.shape != prediction.shape or not np.isfinite([target, prediction]).all():
        raise ValueError("Agreement requires aligned finite observations")
    if not target.size:
        return dict(n=0, r2=None, rmse=None, mae=None, bias=None)
    error = prediction - target
    variance = ((target - target.mean()) ** 2).sum()
    return dict(n=len(target), r2=float(1 - (error**2).sum() / variance) if variance > 0 else None,
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.abs(error).mean()),
                bias=float(error.mean()))


def common_comparison(pairs, predictions):
    """Require all configured models to share the product-valid observation intersection."""
    if not predictions:
        raise ValueError("At least one model prediction table is required")
    if len({r["key"] for r in pairs}) != len(pairs):
        raise ValueError("Duplicate product observation keys")
    for name, rows in predictions.items():
        if len({r["key"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate model observation keys: {name}")
    lookup = {name: {r["key"]: r for r in rows} for name, rows in predictions.items()}
    selected = []
    for pair in pairs:
        if not pair.get("eligible"):
            continue
        key = pair["key"]
        model_values = {}
        for name, rows in lookup.items():
            row = rows.get(key)
            if row is None or not row["valid"] or not np.isfinite(row["prediction"]):
                break
            if not np.isclose(row["original_target"], pair["original_target"],
                              rtol=1e-5, atol=1e-7):
                raise ValueError("Model prediction uses different station labels")
            model_values[name] = row["prediction"]
        if len(model_values) == len(lookup):
            selected.append(dict(pair, predictions=model_values))
    report = {}
    for task in ("sm", "et"):
        rows = [r for r in selected if r["task"] == task]
        values = {"product": [r["product_value"] for r in rows],
                  **{name: [r["predictions"][name] for r in rows] for name in lookup}}
        target = [r["station_value"] for r in rows]
        metrics = {name: agreement(target, y) for name, y in values.items()}
        groups = sorted({r["spatial_group_id"] for r in rows})
        macro = {}
        for name in values:
            by_group = [agreement([r["station_value"] for r in rows
                                   if r["spatial_group_id"] == group],
                                  [r["product_value"] if name == "product" else
                                   r["predictions"][name] for r in rows
                                   if r["spatial_group_id"] == group]) for group in groups]
            macro[name] = {metric: float(np.mean([r[metric] for r in by_group]))
                           if groups else None for metric in ("rmse", "mae", "bias")}
        report[task] = dict(n=len(rows), spatial_groups=len(groups), metrics=metrics,
                            group_macro=macro,
                            product_eligible=sum(r.get("eligible", False) and r["task"] == task
                                                 for r in pairs))
    return report, selected
