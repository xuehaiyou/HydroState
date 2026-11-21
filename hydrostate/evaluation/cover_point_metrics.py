"""Class metrics on human points; no claim of 30 m area-fraction truth."""

import numpy as np

from hydrostate.data.cover_points.contract import CLASSES


def classification_metrics(frame):
    confusion = np.zeros((3, 3), dtype="int64")
    if len(frame):
        truth, prediction = frame.cover_class.to_numpy(), frame.predicted_class.to_numpy()
        if not np.isin(truth, [0, 1, 2]).all() or not np.isin(prediction, [0, 1, 2]).all():
            raise ValueError("Unrecognized cover class")
        np.add.at(confusion, (truth.astype(int), prediction.astype(int)), 1)
    result = dict(
        count=int(confusion.sum()),
        confusion_matrix=confusion.tolist(),
        class_order=list(CLASSES),
        classes={},
    )
    f1s = []
    for i, name in enumerate(CLASSES):
        tp, actual, estimated = (
            int(confusion[i, i]),
            int(confusion[i].sum()),
            int(confusion[:, i].sum()),
        )
        f1 = 2 * tp / (actual + estimated) if actual + estimated else None
        result["classes"][name] = dict(
            support=actual,
            predicted_count=estimated,
            precision=tp / estimated if estimated else None,
            recall=tp / actual if actual else None,
            f1=f1,
            iou=tp / (actual + estimated - tp) if actual + estimated - tp else None,
        )
        if f1 is not None:
            f1s.append(f1)
    result["macro_f1"] = float(np.mean(f1s)) if f1s else None
    result["macro_f1_defined_classes"] = len(f1s)
    result["accuracy"] = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else None
    return result


def summarize(frame):
    eligible = frame.label_valid & frame.prediction_valid & ~frame.training_overlap
    scored = frame[eligible]
    cells = ["sample_id", "observation_date", "output_row", "output_col"]
    # Determine conflict on all valid human labels, before model-specific filtering.
    valid_labels = frame[frame.label_valid]
    conflict = valid_labels.groupby(cells).cover_class.nunique().gt(1)
    conflict_keys = set(conflict[conflict].index)
    consistent = scored[
        [
            tuple(row) not in conflict_keys
            for row in scored[cells].itertuples(index=False, name=None)
        ]
    ]
    unique = consistent.drop_duplicates(cells)
    return dict(
        total_points=len(frame),
        scored_points=len(scored),
        invalid_prediction_points=int((frame.label_valid & ~frame.prediction_valid).sum()),
        training_overlap_points=int(frame.training_overlap.sum()),
        conflicting_cells=len(conflict_keys),
        point_metrics=classification_metrics(scored),
        consistent_unique_cell_metrics=classification_metrics(unique),
        by_site={str(k): classification_metrics(g) for k, g in scored.groupby("site_name")},
        by_site_date={
            f"{k[0]}/{k[1]}": classification_metrics(g)
            for k, g in scored.groupby(["site_name", "observation_date"])
        },
        interpretation="Point classification of an 8-day 30m state; glacier domain only",
    )
