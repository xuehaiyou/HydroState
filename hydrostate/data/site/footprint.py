"""FFP v1.42 on the station grid; absolute mass precedes any crop normalization.

Equations adapted from Kljun et al. (2015), doi:10.5194/gmd-8-3695-2015,
and the authors' Python v1.42. See FFP_LICENSE.txt for the upstream licence.
"""

from math import gamma

import numpy as np
from pyproj import Proj

from .pilot import site_grid

# The fitted crosswind-integrated curve has this analytic whole-plane integral.
# Dividing by it defines probability mass; it is unrelated to tile truncation.
FFP_INTEGRAL = 1.4524 * 1.4622 ** (-0.9914) * gamma(0.9914)


def parameter_valid(zm, canopy, ws, h, ol, sigmav, ustar, wd):
    """Author validity limits plus explicit roughness-sublayer exclusion (z0=.15 hc)."""
    values = np.broadcast_arrays(zm, canopy, ws, h, ol, sigmav, ustar, wd)
    zm, canopy, ws, h, ol, sigmav, ustar, wd = values
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.logical_and.reduce([np.isfinite(v) for v in values])
            & (zm > 0)
            & (canopy >= 0)
            & (zm > 12.5 * 0.15 * canopy)
            & (ws > 0)
            & (h > 10)
            & (zm < h)
            & (ol != 0)
            & (zm / ol > -15.5)
            & (sigmav > 0)
            & (ustar > 0.1)
            & (wd >= 0)
            & (wd <= 360)
        )


def density(east, north, *, zm, ws, h, ol, sigmav, ustar, wd):
    """Probability density in true east/north metres; wind direction is FROM north."""
    angle = np.deg2rad(wd)
    along = east * np.sin(angle) + north * np.cos(angle)
    across = east * np.cos(angle) - north * np.sin(angle)
    scale = zm * (ws / ustar * 0.4) / (1 - zm / h)
    xs = along / scale
    use = xs > 0.1359
    result = np.zeros_like(xs, dtype=float)
    x = xs[use]
    f = 1.4524 * (x - 0.1359) ** -1.9914 * np.exp(-1.4622 / (x - 0.1359)) / scale
    effective_ol = -1e6 if abs(ol) > 5000 else ol
    stability = min(1, 1e-5 * abs(effective_ol / zm) + (0.8 if effective_ol < 0 else 0.55))
    sigma = 2.17 * np.sqrt(1.66 * x**2 / (1 + 20 * x)) / stability * zm * sigmav / ustar
    result[use] = f / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(across[use] ** 2) / (2 * sigma**2))
    return result / FFP_INTEGRAL


def integration_grid(lon, lat, step):
    """Midpoints of the exact 1200 m UTM tile, including convergence and scale."""
    n = round(1200 / step)
    if n % 40 or not np.isclose(n * step, 1200):
        raise ValueError("Integration step must subdivide the 30 m output cells")
    factors = Proj(site_grid(lon, lat)["crs"]).get_factors(lon, lat)
    angle = np.deg2rad(factors.meridian_convergence)
    scale = factors.meridional_scale
    x, y = np.meshgrid(-615 + (np.arange(n) + 0.5) * step, 615 - (np.arange(n) + 0.5) * step)
    east = (x * np.cos(angle) + y * np.sin(angle)) / scale
    north = (-x * np.sin(angle) + y * np.cos(angle)) / scale
    return east, north, (step / scale) ** 2


def window_mass(parameters, durations, lon, lat, *, steps=(10.0, 5.0, 2.5), tolerance=0.01):
    """Duration-average footprints, refining quadrature before approving tile mass.

    Inputs contain only common ET/FFP-valid intervals. A convergence failure remains
    an explicit rejection, never a normalized operator. No LE weighting or smoothing.
    """
    durations = np.asarray(durations, float)
    if len(parameters) != len(durations) or not len(parameters) or np.any(durations <= 0):
        raise ValueError("Positive durations must match footprint parameters")
    previous = None
    difference = float("inf")
    for step in steps:
        east, north, area = integration_grid(lon, lat, step)
        total = np.zeros_like(east)
        for row, duration in zip(parameters, durations, strict=True):
            total += density(east, north, **row) * duration
        n = total.shape[0] // 40
        mass = (total * area / durations.sum()).reshape(40, n, 40, n).sum(axis=(1, 3))
        if previous is not None:
            difference = float(np.abs(mass - previous).sum())
            if difference <= tolerance:
                break
        previous = mass
    captured = float(mass.sum())
    converged = bool(difference <= tolerance and captured <= 1 + tolerance)
    # Use the convergence difference as a conservative numerical margin, not a CI.
    accepted = bool(converged and captured - difference >= 0.95)
    return mass.astype("float32"), dict(
        captured_mass=captured,
        quadrature_l1_difference=difference,
        quadrature_step_m=step,
        converged=converged,
        footprint_tile_valid=accepted,
        truncated_mass=max(0.0, 1 - captured),
        numerical_margin_method="successive_quadrature_L1_not_statistical_CI",
    )
