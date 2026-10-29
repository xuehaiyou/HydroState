"""Physical-domain checks and display-only regional previews."""

from pathlib import Path

import numpy as np
import zarr

from hydrostate.data.gee.contracts import atomic_json

COVER = ("water", "snow_ice", "land")
SURFACES = dict(soil_moisture="land", et_water="water", et_ice="snow_ice", et_land="land")


def check_fields(values, tolerance=0.01):
    """Allow BF16 rounding in fractions; this is not an accuracy evaluation."""
    for task, value in values.items():
        value = np.asarray(value)
        if np.isinf(value).any():
            raise ValueError(f"Infinite regional {task}")
        finite = value[np.isfinite(value)]
        if finite.size and (
            finite.min() < -tolerance
            or (task in (*COVER, "soil_moisture") and finite.max() > 1 + tolerance)
        ):
            raise ValueError(f"Regional {task} outside its physical range")
    fractions = [np.asarray(values[t]) for t in COVER]
    if not all(np.array_equal(np.isfinite(fractions[0]), np.isfinite(v)) for v in fractions[1:]):
        raise ValueError("Coverage masks disagree")
    total = sum(fractions)
    finite = total[np.isfinite(total)]
    if finite.size and np.max(np.abs(finite - 1)) > tolerance:
        raise ValueError("Coverage fractions do not close")


def display_values(values, valid, surface=None, threshold=0.05):
    """Hide negligible surfaces in figures without modifying the product or total ET."""
    mask = np.asarray(valid, dtype=bool)
    if surface is not None:
        mask = mask & np.isfinite(surface) & (surface >= threshold)
    return np.where(mask, values, np.nan)


def render_preview(path, index, *, display_surface_min=0.05):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not 0 <= display_surface_min <= 1:
        raise ValueError("Display surface fraction must be within [0, 1]")
    path = Path(path)
    group = zarr.open_group(str(path), mode="r")
    if not group["date_complete"][index]:
        raise ValueError("Cannot preview an incomplete date")
    bounds = group["time_bounds"][index].astype("datetime64[s]")
    date = str(bounds[0])[:10]
    step = max(1, int(np.ceil(max(group["water"].shape[1:]) / 1200)))
    subset = (index, slice(None, None, step), slice(None, None, step))
    tasks = (*COVER, "soil_moisture", "et_water", "et_ice", "et_land", "et_total")
    fields = {t: group[t][subset] for t in tasks}
    figure = Figure(figsize=(16, 10), layout="constrained")
    FigureCanvasAgg(figure)
    titles = [
        "Water fraction",
        "Snow/ice fraction",
        "Land fraction",
        "Land soil moisture",
        "Water ET",
        "Snow/ice ET",
        "Land ET",
        "Total ET",
    ]
    transform = group.attrs["transform"]
    height, width = group["water"].shape[1:]
    extent = [
        transform[2] / 1000,
        (transform[2] + width * 30) / 1000,
        (transform[5] - height * 30) / 1000,
        transform[5] / 1000,
    ]
    for axis, task, title in zip(figure.subplots(2, 4).flat, tasks, titles):
        surface = fields[SURFACES[task]] if task in SURFACES else None
        value = display_values(
            fields[task], group[task + "_valid"][subset], surface, display_surface_min
        )
        palette, vmax, unit = ("Blues", 1, "fraction")
        if task == "soil_moisture":
            palette, vmax, unit = "YlGnBu", 0.6, "m3/m3"
        elif task.startswith("et_"):
            palette, vmax, unit = "YlOrRd", 10, "mm/day"
        image = axis.imshow(value, origin="upper", extent=extent, cmap=palette, vmin=0, vmax=vmax)
        axis.set(title=title, xlabel="Easting (km)", ylabel="Northing (km)")
        figure.colorbar(
            image, ax=axis, shrink=0.8, label=unit, extend="max" if task not in COVER else "neither"
        )
    figure.suptitle(
        f"HydroState | {date} to {str(bounds[1])[:10]} | 30 m output\n"
        f"Display subsampling: {step}; surface panels shown where fraction >= "
        f"{display_surface_min:g}. Product-supervised retrieval."
    )
    destination = path.parent / f"{path.stem}_previews"
    destination.mkdir(exist_ok=True)
    output = destination / f"{date}.png"
    figure.savefig(output, dpi=180)
    figure.clear()
    atomic_json(
        destination / f"{date}.json",
        dict(
            product=str(path),
            time_index=index,
            display_surface_min=display_surface_min,
            display_stride=step,
            note="Display masks do not change stored conditional rates or total ET",
        ),
    )
    return str(output)
