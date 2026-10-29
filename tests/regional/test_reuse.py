import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from hydrostate.data.gee.contracts import atomic_json, fingerprint
from hydrostate.data.regional import pixels
from hydrostate.data.regional.plan import block_grid, initialize, product
from hydrostate.data.regional.reuse import RegionalReuse
from hydrostate.data.retrieval.contract import INPUT_GROUPS
from hydrostate.storage.regional import commit, normalize_raw, product_path


def plans(tmp_path):
    cfg = json.loads(Path("configs/inference/poyang_pilot.json").read_text())
    source = initialize(tmp_path / "old", cfg)
    source["grid"].update(height=240, width=240)
    source.pop("identity")
    source["identity"] = fingerprint(source)
    atomic_json(tmp_path / "old/plan.json", source)
    target = copy.deepcopy(source)
    target["grid"]["left"] += 600
    target["grid"]["top"] -= 600
    target["grid"].update(height=120, width=120)
    target.pop("identity")
    target["identity"] = fingerprint(target)
    return source, target, cfg["pixels"]


@pytest.mark.parametrize("name", list(INPUT_GROUPS))
def test_reuse_shifted_grid_all_groups_preserves_values_masks_and_time(tmp_path, monkeypatch, name):
    source, target, settings = plans(tmp_path)
    spec = INPUT_GROUPS[name]
    orbit = [25, "ASCENDING"] if spec.sensor == "s1" else None
    slot = source["periods"][0]["slots"][0]
    r = spec.resolution
    yy, xx = np.indices((2400 // r, 2400 // r))
    raw = np.zeros((len(spec.bands) + 4, *xx.shape), "f4")
    raw[: len(spec.bands)] = (xx + yy * 3)[None] / 100
    raw[-4:] = np.array([2, 3, 1, 5])[:, None, None]
    raw[:, 700 // r : 900 // r, 700 // r : 900 // r] = 0
    for y in range(2):
        for x in range(2):
            old = product(source, name, slot, [y, x], orbit)
            n = 1200 // r
            commit(tmp_path / "old", source, old, raw[:, y * n : (y + 1) * n, x * n : (x + 1) * n])
    item = product(target, name, slot, [0, 0], orbit)
    reuse = RegionalReuse(target, [tmp_path / "old"])
    result = reuse.read(item)
    expected = normalize_raw(target, name, raw[:, 600 // r : 1800 // r, 600 // r : 1800 // r])
    for field in expected:
        np.testing.assert_array_equal(result[0][field], expected[field])
    assert len(result[1]) == 4
    monkeypatch.setattr(
        pixels, "compute_pixels", lambda *a, **kw: pytest.fail("duplicate download")
    )
    job = dict(
        id="fixture",
        group=name,
        products=[item],
        grid=block_grid(target, [0, 0], r),
        scenes=[["scene"]],
        bands=["t0_" + b for b in (*spec.bands, "days", "mean_day", "first_day", "last_day")],
        raw_bytes=(len(spec.bands) + 4) * (1200 // r) ** 2 * 4,
    )
    receipt = pixels.receive(None, tmp_path / "new", target, job, settings, reuse)
    assert receipt["network_requests"] == 0 and receipt["reused_products"] == 1
    group = zarr.open_group(str(product_path(tmp_path / "new", item)), mode="r")
    for field in expected:
        np.testing.assert_array_equal(group[field][:], expected[field])
    assert (product_path(tmp_path / "new", item) / "_REUSE.json").exists()
    wrong_date = product(target, name, source["periods"][0]["slots"][1], [0, 0], orbit)
    assert reuse.read(wrong_date) is None
    if orbit:
        assert reuse.read(product(target, name, slot, [0, 0], [26, "ASCENDING"])) is None


def test_partial_coverage_downloads_one_whole_block_and_rejects_incompatible_sources(
    tmp_path, monkeypatch
):
    source, target, settings = plans(tmp_path)
    slot = source["periods"][0]["slots"][0]
    old = product(source, "s1_10m", slot, [0, 0], [25, "ASCENDING"])
    commit(tmp_path / "old", source, old, np.zeros((6, 120, 120), "f4"))
    item = product(target, "s1_10m", slot, [0, 0], [25, "ASCENDING"])
    reuse = RegionalReuse(target, [tmp_path / "old"])
    assert reuse.read(item) is None
    bands = ["t0_" + b for b in ("VV", "VH", "days", "mean_day", "first_day", "last_day")]
    job = dict(
        id="fixture",
        group="s1_10m",
        products=[item],
        grid=block_grid(target, [0, 0]),
        scenes=[["scene"]],
        bands=bands,
        raw_bytes=6 * 120 * 120 * 4,
    )
    calls = []
    monkeypatch.setattr(pixels, "composite_image", lambda ee, job: job)

    def fetch(ee, image, names, crs, transform, width, height, **kwargs):
        calls.append((width, height, transform))
        return np.zeros((height, width), dtype=[(b, "f4") for b in names])

    monkeypatch.setattr(pixels, "compute_pixels", fetch)
    receipt = pixels.receive(None, tmp_path / "new", target, job, settings, reuse)
    assert calls == [(120, 120, job["grid"]["transform"])]
    assert receipt["reused_products"] == 0 and receipt["network_requests"] == 1
    incompatible = copy.deepcopy(target)
    incompatible["grid"]["left"] += 10
    with pytest.raises(ValueError, match="aligned"):
        RegionalReuse(incompatible, [tmp_path / "old"])
    incompatible = copy.deepcopy(target)
    incompatible["normalization"]["extra"] = "changed"
    with pytest.raises(ValueError, match="normalization"):
        RegionalReuse(incompatible, [tmp_path / "old"])
