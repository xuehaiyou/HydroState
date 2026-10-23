"""Align equal-extent native composites at read time, without inventing observations."""

import numpy as np

from hydrostate.data.gee.products import BANDS

from .contract import INPUT_GROUPS, INPUT_SIZE, SENSORS, RetrievalSchema


def nearest_grid(values, shape):
    """Pixel-centre nearest neighbour, including the noninteger 15 -> 10 m ratio.

    All grids share outer bounds. Exact boundary ties select the higher index.
    Using the same indices for spectra, masks and dates preserves their support.
    """
    if values.shape[-2:] == tuple(shape):
        return values
    indices = [
        ((2 * np.arange(dst) + 1) * src) // (2 * dst) for src, dst in zip(values.shape[-2:], shape)
    ]
    return values[..., indices[0][:, None], indices[1]]


def sensor_masks(inputs, shape=(INPUT_SIZE, INPUT_SIZE)):
    """Conservative all-band validity on the eventual model grid."""
    return {
        sensor: np.logical_and.reduce(
            [
                nearest_grid(inputs[f"{name}_valid"], shape).astype(bool)
                for name, group in INPUT_GROUPS.items()
                if group.sensor == sensor
            ]
        )
        for sensor in SENSORS
    }


def align_inputs(inputs, schema=None):
    schema = schema or RetrievalSchema()
    shape = (schema.height, schema.width)
    result = {key: inputs[key] for key in schema.input_metadata}
    masks = sensor_masks(inputs, shape)
    for sensor, bands in BANDS.items():
        channels = {}
        for name, group in INPUT_GROUPS.items():
            if group.sensor != sensor:
                continue
            values = nearest_grid(inputs[name], shape)
            channels.update({band: values[:, index] for index, band in enumerate(group.bands)})
            for suffix in ("valid", "quality"):
                key = f"{name}_{suffix}"
                result[key] = nearest_grid(inputs[key], shape)
        result[sensor] = np.stack([channels[band] for band in bands], axis=1)
        result[sensor] *= masks[sensor]
        result[f"{sensor}_valid"] = masks[sensor].astype(np.uint8)
    return result


def grid_metadata(sample):
    return {
        name: dict(
            sensor=group.sensor,
            bands=list(group.bands),
            resolution_m=group.resolution,
            crs=sample["crs"],
            transform=group.transform(sample["transform"]),
            shape=list(group.shape()),
        )
        for name, group in INPUT_GROUPS.items()
    }
