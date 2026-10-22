"""Configuration-selected regional acquisition with lazy backend imports."""

import json
import logging
from pathlib import Path

from hydrostate.data.input_products import L2_PROFILE, LEGACY_PROFILE
from hydrostate.data.regional.temporal import policy

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs/experiments/legacy/data"
BACKENDS = ("drive", "pixels", "local")


def load_config(path=None, *, backend=None, download_config=None, local_config=None):
    """Resolve the canonical config or preserve explicit legacy CLI selections."""
    if path is not None:
        if any(v is not None for v in (backend, download_config, local_config)):
            raise ValueError("--input-config cannot be combined with legacy backend/config flags")
        config = json.loads(Path(path).read_text())
        if set(config) != {"backend", "input_profile", "options"}:
            raise ValueError("Input config requires backend, input_profile and options")
    else:
        backend = backend or ("drive" if download_config else "local" if local_config else "pixels")
        if download_config and backend != "drive" or local_config and backend != "local":
            raise ValueError("Backend conflicts with the supplied transport config")
        options = {}
        if backend == "drive":
            source = Path(download_config or CONFIG_ROOT / "gee_drive.json")
            options = json.loads(source.read_text())
        elif backend == "local":
            source = Path(local_config or CONFIG_ROOT / "hpc_local_l2.json")
            options = json.loads(source.read_text())
        config = dict(
            backend=backend,
            input_profile=L2_PROFILE if backend == "local" else LEGACY_PROFILE,
            options=options,
        )
    if config["backend"] not in BACKENDS or not isinstance(config["options"], dict):
        raise ValueError("Invalid regional backend/options")
    expected = L2_PROFILE if config["backend"] == "local" else LEGACY_PROFILE
    if config["input_profile"] != expected:
        raise ValueError(f"{config['backend']} currently requires input_profile={expected}")
    if config["backend"] == "pixels" and config["options"]:
        raise ValueError("Direct pixels use --workers; no transport options are supported")
    return config


def prepare_region(
    root, plan, periods, config, *, workers=4, min_free_gib=20, drive_token=None, quota_project=None
):
    """All backends publish the same offline manifests/cache, within their profile."""
    profile = plan["config"].get("input_profile", LEGACY_PROFILE)
    sampling = policy(plan["config"])
    if config["input_profile"] != profile:
        raise ValueError(
            f"Input config profile {config['input_profile']} differs from frozen plan {profile}; "
            "initialize a separate regional root"
        )
    backend = config["backend"]
    if backend != "drive" and (drive_token is not None or quota_project is not None):
        raise ValueError("Drive credentials/options require the Drive backend")
    logging.getLogger(__name__).info(
        "Regional inputs: backend=%s profile=%s temporal=%s root=%s",
        backend, profile, sampling, root
    )
    kwargs = dict(workers=workers, min_free_gib=min_free_gib)
    if backend == "drive":
        from .gee_drive import prepare_region_drive

        return prepare_region_drive(
            root,
            plan,
            periods,
            config["options"],
            **kwargs,
            drive_token=drive_token,
            quota_project=quota_project,
        )
    if backend == "local":
        from .local import prepare_region_local

        return prepare_region_local(root, plan, periods, config["options"], **kwargs)
    if backend == "pixels":
        from .gee_pixels import prepare_date

        for period in periods:
            prepare_date(root, plan, period, **kwargs)
        return
    raise ValueError(f"Unknown regional backend: {backend}")
