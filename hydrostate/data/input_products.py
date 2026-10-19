"""Versioned input semantics shared by GEE sample and local regional producers."""

import numpy as np

LEGACY_PROFILE = "olmoearth_l1_v1"
L2_PROFILE = "landsat_l2_sr_v1"
LANDSAT_SR_BANDS = [f"SR_B{i}" for i in range(1, 8)]
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


def profile_bands(profile):
    from hydrostate.data.gee.products import BANDS

    if profile not in {LEGACY_PROFILE, L2_PROFILE}:
        raise ValueError(f"Unknown input profile: {profile}")
    return dict(BANDS, landsat=LANDSAT_SR_BANDS) if profile == L2_PROFILE else BANDS


def landsat_l2_pixels(dn, qa_pixel, qa_radsat, valid=None):
    """Do not clip negative water reflectance or mask the QA water bit."""
    dn = np.asarray(dn)
    mask = ((qa_pixel.astype('uint16') & 63) == 0) & (qa_radsat == 0)
    mask &= (dn != 0).all(axis=0) & np.isfinite(dn).all(axis=0)
    if valid is not None:
        mask &= valid
    sr = dn.astype('float32') * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    return np.where(mask[None], sr, 0), mask


def landsat_l2_spec():
    return dict(shape=[4, 7, 120, 120], dtype="float32", units="1", grid="input",
                temporal="sequence", bands=LANDSAT_SR_BANDS, input_profile=L2_PROFILE,
                normalization="physical_surface_reflectance", resampling="nearest",
                native_resolution_m=30)


def harmonize_s2(dn, *, offset, quantification):
    """SAFE BOA DN -> the nonnegative 10000-scaled SR used by GEE Harmonized.

    Callers retain the original DN=0 mask before adding the metadata offset.
    Baseline is determined from metadata, never from acquisition year.
    """
    if not np.isfinite(quantification) or quantification <= 0:
        raise ValueError("Invalid S2 BOA quantification value")
    return np.maximum(0, (dn.astype('float32') + offset) * (10000 / quantification))
