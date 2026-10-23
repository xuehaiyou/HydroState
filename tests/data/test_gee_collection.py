"""Current collector lifecycle: hard quotas only count validated committed shards."""

import json
from types import SimpleNamespace

import pytest

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.retrieval import collector
from hydrostate.data.retrieval.cli import DEFAULT_CONFIG, resolve_config
from hydrostate.data.retrieval.contract import PROFILE, VERSION


def config(tmp_path, samples=8):
    return resolve_config(
        SimpleNamespace(config=DEFAULT_CONFIG, output=tmp_path, samples=samples, project=None)
    )


def test_current_config_and_dry_run(tmp_path):
    cfg = config(tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert cfg["start_year"] == 2017 and cfg["max_dates_per_location"] == 4
    assert cfg["min_date_gap_days"] == 40 and cfg["min_known_fraction"] == 0.95
    assert cfg["water_confidence"] == 0.35 and cfg["ice_confidence"] == 0.60
    assert cfg["land_confidence"] == 0.85 and cfg["land_max_water_ice_probability"] == 0.10
    with pytest.raises(ValueError, match="eight"):
        config(tmp_path, 2)


def test_refuses_existing_dataset_before_remote_calls(tmp_path, monkeypatch):
    (tmp_path / "old.parquet").touch()
    monkeypatch.setattr(collector, "initialize", lambda *_: pytest.fail("remote call"))
    with pytest.raises(ValueError, match="existing dataset"):
        collector.collect(config(tmp_path))


def test_changed_contract_cannot_resume(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    atomic_json(tmp_path / "collection.json", dict(cfg, version="old", input_profile=PROFILE))
    monkeypatch.setattr(collector, "initialize", lambda *_: pytest.fail("remote call"))
    with pytest.raises(ValueError, match="contract changed"):
        collector.collect(cfg)


@pytest.mark.parametrize("samples,max_dates", [(8, 1), (16, 4)])
def test_quota_refill_and_resume(tmp_path, monkeypatch, samples, max_dates):
    import olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize as normalizer

    cfg = config(tmp_path, samples)
    cfg["max_dates_per_location"] = max_dates
    monkeypatch.setattr(collector, "initialize", lambda *_: object())
    monkeypatch.setattr(normalizer, "load_computed_config", lambda: {})
    monkeypatch.setattr(
        collector,
        "pml_calendar",
        lambda *a: {"2020-01-01": {}, "2020-03-01": {}, "2021-01-01": {}, "2021-03-01": {}},
    )
    candidates = [
        dict(
            crs="EPSG:32650",
            transform=[10, 0, i * 2000, 0, -10, 10000],
            lon=117.0,
            lat=27.0,
            frequency_bin=i % 7,
            sampling_group=i % 7,
            climate_class=1,
            block=[116, 26, 2],
            split="train",
        )
        for i in range(samples + 7)
    ]
    monkeypatch.setattr(
        collector, "candidate_blocks", lambda *a: iter([candidates[:7], candidates[7:]])
    )

    class FakeIce:
        def __init__(self, *args):
            self.stats = {}

        def next_query(self, *args):
            return {"target_time": "2020-03-01"}

        def fetch(self, query):
            return [
                dict(
                    candidates[0],
                    sampling_group=7,
                    frequency_bin=-1,
                    candidate_target_time=query["target_time"],
                    transform=[10, 0, (100 + i) * 2000, 0, -10, 10000],
                )
                for i in range(8)
            ]

        def observe(self, row):
            pass

    monkeypatch.setattr(collector, "IceCandidateSource", FakeIce)
    attempts = []

    def collect(ee, sample, root, *args):
        attempts.append(sample)
        if sample["transform"][2] == 0:  # Reject the first zero-frequency centre.
            return None
        row = dict(
            sample,
            sample_id=str(sample["transform"][2]),
            version=VERSION,
            input_profile=PROFILE,
            target_end="2020-01-09",
            shard_index=0,
            input_path="inputs.zarr",
            label_path="labels.zarr",
        )
        (root / "records").mkdir(exist_ok=True)
        atomic_json(root / "records" / f"{row['sample_id']}.json", row)
        return row

    monkeypatch.setattr(collector, "collect_sample", collect)
    collector.collect(cfg)
    report = json.loads((tmp_path / "collection_report.json").read_text())
    assert report["complete"] and report["samples"] == samples
    assert list(report["sampling_group_counts"].values()) == [samples // 8] * 8
    assert report["unique_locations"] == samples  # Fresh locations precede repeated dates.
    assert len(attempts) == samples + sum(row["transform"][2] == 0 for row in attempts)
    assert sum(report["year_month_counts"].values()) == samples
    assert sum(report["split_counts"].values()) == samples
    monkeypatch.setattr(
        collector, "initialize", lambda *_: pytest.fail("remote call on complete resume")
    )
    collector.collect(cfg)
