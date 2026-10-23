"""Array-level checks shared by collection and release audits."""

import numpy as np

from .alignment import sensor_masks
from .contract import COVER, ET, INPUT_GROUPS, SENSORS, RetrievalSchema


def validate_arrays(inputs, targets):
    schema = RetrievalSchema()
    for values, specs in ((inputs, schema.inputs), (targets, schema.targets)):
        if set(values) != set(specs):
            raise ValueError("Array names disagree with retrieval schema")
        for name, spec in specs.items():
            if values[name].shape != spec.tail_shape or not np.isfinite(values[name]).all():
                raise ValueError(f"Invalid array {name}")
    for name in INPUT_GROUPS:
        if np.any(inputs[f"{name}_quality"][:, :1] < 0):
            raise ValueError("Negative observation count")
        if not np.array_equal(inputs[f"{name}_quality"][:, :1] > 0, inputs[f"{name}_valid"] > 0):
            raise ValueError("Observation count and validity disagree")
    masks = sensor_masks(inputs, (schema.height, schema.width))
    presence = np.stack([masks[s].any(axis=(1, 2, 3)) for s in SENSORS], axis=1)
    if not np.array_equal(presence, inputs["sensor_valid"] > 0):
        raise ValueError("Sensor availability disagrees with aligned native masks")
    known = targets["cover_known"] > 0
    if not np.allclose(sum(targets[name] for name in COVER)[known], 1, atol=1e-5):
        raise ValueError("Cover fractions do not close")
    for task in ET:
        weights = targets[f"{task}_native_weights"]
        valid = targets[f"{task}_native_valid"].astype(bool)
        if (weights < 0).any() or not np.allclose(weights.sum((-2, -1))[valid], 1, atol=1e-5):
            raise ValueError("Invalid native weights")
