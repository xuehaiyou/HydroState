"""Config isolation and routing for the three regional acquisition paths."""

from pathlib import Path

import pytest

from hydrostate.data.regional.prepare import load_config, prepare_region

CONFIG = Path("configs/data/regional")


@pytest.mark.parametrize(
    "name,backend",
    [
        ("gee_drive", "drive"),
        ("gee_pixels", "pixels"),
        ("local", "local"),
    ],
)
def test_canonical_configs(name, backend):
    config = load_config(CONFIG / f"{name}.json")
    assert config["backend"] == backend
    with pytest.raises(ValueError, match="cannot be combined"):
        load_config(CONFIG / f"{name}.json", backend=backend)


def test_legacy_cli_preserves_previous_configs():
    assert load_config()["backend"] == "pixels"
    drive = load_config(backend="drive")
    assert "export_layout" not in drive["options"]
    assert (
        load_config(download_config="configs/data/gee_drive_native.json")["options"][
            "export_layout"
        ]
        == "native_v1"
    )
    with pytest.raises(ValueError, match="conflicts"):
        load_config(backend="pixels", download_config="configs/data/gee_drive.json")


def test_profile_mismatch_rejected_before_any_backend_runs(tmp_path):
    config = load_config(CONFIG / "local.json")
    with pytest.raises(ValueError, match="separate regional root"):
        prepare_region(tmp_path, dict(config={}), [], config)


def test_local_config_dispatches_without_initializing_gee(tmp_path, monkeypatch):
    from hydrostate.data.regional import local

    config = load_config(CONFIG / "local.json")
    plan = dict(config=dict(input_profile=config["input_profile"]))
    calls = []
    monkeypatch.setattr(local, "prepare_region_local", lambda *a, **k: calls.append((a, k)))
    prepare_region(tmp_path, plan, [], config, workers=2)
    assert len(calls) == 1 and calls[0][0][3] == config["options"]
    assert calls[0][1]["workers"] == 2
