import json

import numpy as np
import zarr

from hydrostate.storage import SampleShardWriter, Schema


def test_sample_shard_writer(tmp_path):
    schema = Schema(time_steps=2, height=8, width=8)
    input_path = tmp_path / "inputs.zarr"
    label_path = tmp_path / "labels.zarr"
    writer = SampleShardWriter(str(input_path), str(label_path), 1, schema=schema)
    inputs = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in schema.inputs.items()
    }
    targets = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in schema.targets.items()
    }
    assert writer.write(inputs, targets) == 0
    writer.finalize()
    assert json.loads((input_path / "_SUCCESS.json").read_text())["complete"]
    assert zarr.open_group(str(input_path), mode="r")["s2"].shape == (1, 2, 12, 8, 8)


def test_gee_sampling_contract():
    from datetime import date

    from hydrostate.data.gee_sampling import climate_quota, frequency_bin, slots

    assert [frequency_bin(v) for v in [0, 1, 20, 21, 40, 60, 80, 99, 100]] == [
        0,
        1,
        1,
        2,
        2,
        3,
        4,
        5,
        6,
    ]
    windows = slots("2020-03-01")
    assert windows[0][0] == "2019-12-29"
    assert windows[-1][1] == "2020-03-02"
    for a, b in windows:
        assert (date.fromisoformat(b) - date.fromisoformat(a)).days == 16
    assert all(windows[i][1] == windows[i + 1][0] for i in range(3))
    allocation = climate_quota(100, {1: 90, 2: 10})
    assert allocation == {1: 70, 2: 30}
    assert Schema().inputs["s2"].tail_shape == (4, 12, 120, 120)


def test_30m_supervision():
    import torch

    from hydrostate.models.data_preprocessor import aggregate_targets

    water = torch.zeros(1, 1, 120, 120)
    water[0, 0, 0, :3] = 1
    valid = torch.ones_like(water, dtype=torch.bool)
    valid[0, 0, 4, 4] = False
    gradient = torch.arange(14400).reshape(1, 1, 120, 120).float()
    targets = dict(
        water=water,
        water_valid=valid,
        soil_moisture=gradient.clone(),
        soil_moisture_valid=valid.clone(),
        et=gradient.clone(),
        et_valid=valid.clone(),
    )
    aggregate_targets(targets)
    assert targets["water"].shape == (1, 1, 40, 40)
    assert torch.isclose(targets["water"][0, 0, 0, 0], torch.tensor(1 / 3))
    assert not targets["water_valid"][0, 0, 1, 1]
    assert targets["water_valid"][0, 0, 0, 0]
    assert targets["et"].shape == gradient.shape  # ET uses separate native weights.
    assert targets["soil_moisture"].shape == (1, 1, 1, 1)
    assert targets["soil_moisture"].item() == gradient[0, 0, 60, 60]


def test_native_pml_geometry_clips_partial_cells_and_weights_boundaries():
    from hydrostate.data.gee_sampling import pml_native_targets

    sample = dict(
        lon=0.0,
        lat=0.0,
        crs="EPSG:6933",
        transform=[10, 0, 0, 0, -10, 120],
        labels={"pml": {"projection": {"crs": "EPSG:6933", "transform": [50, 0, 5, 0, -50, 115]}}},
    )
    native = pml_native_targets(sample, np.full((12, 12), 4.0), np.ones((12, 12), bool))
    assert native["weights"].shape == (4, 4, 4)  # Only four full cells fit.
    np.testing.assert_allclose(native["weights"].sum(axis=(1, 2)), 1.0)
    np.testing.assert_allclose(native["value"], 4.0)
    # Native cell [5,55] crosses two output columns with equal 25 m overlap.
    np.testing.assert_allclose(native["weights"][0, :2, :2], 0.25, atol=1e-6)
    assert not native["weights"][0, 2:, :].any()


def test_native_collate_and_flip_are_consistent():
    import torch

    from hydrostate.datasets.zarr_dataset import HydroStateZarrDataset, hydro_collate

    samples = []
    for n in (1, 3):
        samples.append(
            dict(
                inputs={"s1": torch.zeros(4, 2, 12, 12)},
                data_samples={
                    "et_native_weights": torch.arange(n * 16).reshape(n, 4, 4).float(),
                    "et_native_value": torch.ones(n),
                    "et_native_valid": torch.ones(n, dtype=torch.bool),
                },
            )
        )
    dataset = object.__new__(HydroStateZarrDataset)
    dataset.training, dataset.flip_probability = True, 1.0
    expected = samples[0]["data_samples"]["et_native_weights"].flip((-1, -2))
    dataset._augment(samples[0])
    torch.testing.assert_close(samples[0]["data_samples"]["et_native_weights"], expected)
    batch = hydro_collate(samples)["data_samples"]
    assert batch["et_native_weights"].shape == (2, 3, 4, 4)
    assert batch["et_native_valid"].tolist() == [[True, False, False], [True, True, True]]


def test_gee_histogram_excludes_null_ocean_blocks():
    from hydrostate.data.gee_sampling import stratum_counts

    assert stratum_counts({'null': 28459592}) == {}
    assert stratum_counts(None) == {}
    assert stratum_counts({}) == {}
    assert stratum_counts({
        'null': 100, '11': 3, '11.0': 4, '25': 9,
        'NaN': 5, 'inf': 5, '11.5': 5, '9': 5, '17': 5, '0': 5,
        '31': None, '32': 0, '33': -1, '34': float('inf'),
    }) == {11: 7, 25: 9}


def test_seven_frequency_bins_and_balanced_quotas():
    from hydrostate.data.gee_sampling import FREQUENCY_BINS, frequency_bin, quota, stratum_counts

    assert [frequency_bin(v) for v in [None, -1, 101, float('nan')]] == [None] * 4
    assert frequency_bin(0.5) == 1
    assert frequency_bin(99.5) == 5
    counts = stratum_counts({'10': 12, '16': 9, '11': 4, 'null': 99})
    assert counts == {10: 12, 16: 9, 11: 4}
    allocation = quota(10000, FREQUENCY_BINS)
    assert set(allocation) == set(range(7))
    assert sum(allocation.values()) == 10000
    assert max(allocation.values()) - min(allocation.values()) == 1


def test_climate_screen_empty_ocean_and_coastal_land():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hydrostate.data.gee_sampling import screen_block

    ee = SimpleNamespace(Reducer=SimpleNamespace(max=lambda: Mock()))
    kg, area = Mock(), Mock()
    eligible = kg.gt.return_value.rename.return_value
    result = eligible.reduceRegion.return_value.getInfo
    for value, expected in [(None, False), (0, False), (1, True)]:
        result.return_value = {'eligible': value}
        assert screen_block(ee, kg, area) is expected
    kwargs = eligible.reduceRegion.call_args.kwargs
    assert kwargs['scale'] == kg.projection().nominalScale()
    assert kwargs['crs'] == kg.projection()
    area.buffer.assert_called_with(2000)


def test_shared_candidate_pool_reuses_blocks_for_other_sample_seeds(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import hydrostate.data.gee_sampling as sampling

    cache = tmp_path / 'shared'
    cache.mkdir()
    expected = dict(sampling_strategy=sampling.SAMPLING_STRATEGY, climate_asset='climate',
                    climate_band=None, candidate_seed=42, candidates_per_class=16)
    sampling.check_candidate_config(cache, expected)
    block = {'features': [{'properties': {'stratum': 10},
                           'geometry': {'coordinates': [1, 2]}}], 'counts': {'10': 100}}
    sampling.atomic_json(cache / '0000.json', block)
    # Restrict this offline fixture to one cached block; no EE client is provided.
    monkeypatch.setattr(sampling, 'range', lambda *a: [-60] if a[0] == -60 else [-180],
                        raising=False)
    args = SimpleNamespace(candidate_dir=cache, seed=42, samples=10000,
                           **{k: v for k, v in expected.items() if k != 'sampling_strategy'})
    first, counts = sampling.candidate_pool(None, args, tmp_path / 'run1', None, None, None)
    args.seed, args.samples = 99, 30000
    second, again = sampling.candidate_pool(None, args, tmp_path / 'run2', None, None, None)
    assert counts == again == {10: 100}
    assert first[0]['geometry'] == second[0]['geometry']
    assert first[0]['priority'] != second[0]['priority']
    assert json.loads((cache / '0000.json').read_text()) == block
    import pytest
    with pytest.raises(RuntimeError, match='configuration differs'):
        sampling.check_candidate_config(cache, dict(expected, candidate_seed=99))


def test_shared_pool_refuses_to_race_running_original_writer(tmp_path):
    import fcntl
    from types import SimpleNamespace
    import pytest
    from hydrostate.data.gee_sampling import candidate_pool

    source = tmp_path / 'original'
    cache = source / 'candidates'
    cache.mkdir(parents=True)
    (source / 'config.json').write_text('{}')
    args = SimpleNamespace(candidate_dir=cache, climate_asset='climate', climate_band=None,
                           candidate_seed=42, candidates_per_class=16)
    with (source / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='still owns'):
            candidate_pool(None, args, tmp_path / 'new', None, None, None)
    assert not (cache / 'pool_config.json').exists()


def test_completed_original_pool_can_be_read_while_run_exports(tmp_path, monkeypatch):
    import fcntl
    from types import SimpleNamespace
    import hydrostate.data.gee_sampling as sampling

    source = tmp_path / 'original'
    cache = source / 'candidates'
    cache.mkdir(parents=True)
    metadata = dict(sampling_strategy=sampling.SAMPLING_STRATEGY, climate_asset='climate',
                    climate_band=None, candidate_seed=42, candidates_per_class=16)
    sampling.atomic_json(source / 'config.json', metadata)
    sampling.atomic_json(cache / '0000.json', {'features': [], 'counts': {}, 'screened_out': True})
    def one_block(*args):
        return [0] if args == (12960,) else ([-60] if args[0] == -60 else [-180])
    monkeypatch.setattr(sampling, 'range', one_block, raising=False)
    args = SimpleNamespace(candidate_dir=cache, seed=42,
                           **{k: v for k, v in metadata.items() if k != 'sampling_strategy'})
    with (source / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sampling.candidate_pool(None, args, tmp_path / 'next_run', None, None, None) == ([], {})
