"""A recency revision retains observations but requires newly admitted inputs to be ready."""

import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hydrostate.data.gee.contracts import atomic_json, fingerprint, locked
from tools.data import update_region_recency as revision


@pytest.fixture
def root(tmp_path):
    config = dict(latest_days=3, min_active_slots=2, min_valid_fraction=0.5)
    plan = dict(config=config, identity=dict(config=copy.deepcopy(config)), grid=dict(crs="fixed"))
    plan["plan_hash"] = fingerprint(plan)
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(
        tmp_path / "selection_contract.json",
        dict(plan=fingerprint(plan), preprocessing="unchanged", selection_version=1),
    )
    with sqlite3.connect(tmp_path / "preparation.sqlite") as db:
        db.execute("CREATE TABLE windows (date TEXT, id TEXT, result TEXT, PRIMARY KEY(date,id))")
        for day, cases in [
            ("2017-04-11", [(2.5, 4), (3.57, 4), (4.00001, 4), (None, 0), (1.0, 1)]),
            ("2017-04-19", [(2.0, 4)]),
        ]:
            rows = []
            for i, (age, active) in enumerate(cases):
                row = dict(
                    window_id=str(i),
                    time=day,
                    latest_age_days=age,
                    active_slots=active,
                    accepted=active >= 2 and age is not None and age <= 3,
                    selections=json.dumps(dict(asset_id="preserved")),
                )
                db.execute("INSERT INTO windows VALUES (?,?,?)", (day, str(i), json.dumps(row)))
                rows.append(row)
            for folder in ("manifests", "selections"):
                (tmp_path / folder).mkdir(exist_ok=True)
                pd.DataFrame(rows).to_parquet(tmp_path / folder / f"{day}.parquet")
            (tmp_path / "reports").mkdir(exist_ok=True)
            atomic_json(tmp_path / "reports" / f"preparation_{day}.json", dict(status="ready"))
    (tmp_path / "exports.sqlite").write_bytes(b"existing task IDs")
    (tmp_path / "input_cache").mkdir()
    (tmp_path / "input_cache" / "cached-data").write_bytes(b"existing pixels")
    return tmp_path


def test_revision_preserves_assets_and_invalidates_only_changed_dates(root):
    original_manifest = (root / "manifests/2017-04-19.parquet").read_bytes()
    report = revision.update_recency(root, 4)
    assert report["dates"][0]["accepted_before"] == 1
    assert report["dates"][0]["accepted_after"] == 2
    with sqlite3.connect(root / "preparation.sqlite") as db:
        rows = [
            json.loads(row[0]) for row in db.execute("SELECT result FROM windows ORDER BY date,id")
        ]
    assert [r["accepted"] for r in rows] == [True, True, False, False, False, True]
    assert all(json.loads(r["selections"])["asset_id"] == "preserved" for r in rows)
    assert not (root / "manifests/2017-04-11.parquet").exists()
    assert not (root / "selections/2017-04-11.parquet").exists()
    assert not (root / "reports/preparation_2017-04-11.json").exists()
    assert (root / "manifests/2017-04-19.parquet").read_bytes() == original_manifest
    plan = json.loads((root / "plan.json").read_text())
    contract = json.loads((root / "selection_contract.json").read_text())
    assert plan["config"]["latest_days"] == plan["identity"]["config"]["latest_days"] == 4
    assert plan["config"]["min_valid_fraction"] == 0.5
    assert contract["preprocessing"] == "unchanged" and contract["plan"] == fingerprint(plan)
    assert plan["plan_hash"] == fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
    assert (root / "exports.sqlite").read_bytes() == b"existing task IDs"
    assert (root / "input_cache/cached-data").read_bytes() == b"existing pixels"
    assert not (root / revision.PENDING).exists()
    backup = root / "revisions" / next((root / "revisions").iterdir()).name
    assert (backup / "manifests/2017-04-11.parquet").exists()
    with sqlite3.connect(backup / "preparation.sqlite") as db:
        assert (
            json.loads(
                db.execute(
                    "SELECT result FROM windows WHERE date=? AND id='1'", ("2017-04-11",)
                ).fetchone()[0]
            )["accepted"]
            is False
        )
    assert revision.update_recency(root, 4) == dict(changed=False, latest_days=4)


def test_interrupted_revision_resumes_without_reselecting(root, monkeypatch):
    original = revision.atomic_json

    def interrupt(path, value):
        if path == root / "plan.json":
            raise OSError("interrupted publication")
        original(path, value)

    monkeypatch.setattr(revision, "atomic_json", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        revision.update_recency(root, 4)
    assert (root / revision.PENDING).exists()
    cli = subprocess.run(
        [
            sys.executable,
            str(Path(revision.__file__).parents[1] / "predict_region.py"),
            "prepare",
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 2 and "Interrupted recency revision" in cli.stderr
    monkeypatch.setattr(revision, "atomic_json", original)
    report = revision.update_recency(root, 4)
    assert report["dates"][0]["accepted_after"] == 2
    assert not (root / revision.PENDING).exists()


def test_revision_refuses_running_writer_and_existing_predictions(root):
    with locked(root / "prepare.lock", blocking=False):
        with pytest.raises(BlockingIOError):
            revision.update_recency(root, 4)
    (root / "products/poyang_3vars.zarr").mkdir(parents=True)
    with pytest.raises(ValueError, match="Existing predictions"):
        revision.update_recency(root, 4)
    assert json.loads((root / "plan.json").read_text())["config"]["latest_days"] == 3


def test_disable_recency_preserves_age_and_rejects_empty_or_single_slot(root):
    report = revision.update_recency(root, None)
    assert report["dates"][0]["accepted_after"] == 3
    with sqlite3.connect(root / "preparation.sqlite") as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT result FROM windows ORDER BY date,id")]
    assert [r["accepted"] for r in rows] == [True, True, True, False, False, True]
    assert rows[2]["latest_age_days"] == 4.00001
    assert json.loads((root / "plan.json").read_text())["config"]["latest_days"] is None
    assert revision.update_recency(root, None) == dict(changed=False, latest_days=None)
    with pytest.raises(ValueError, match="only relaxes"):
        revision.update_recency(root, 15)
