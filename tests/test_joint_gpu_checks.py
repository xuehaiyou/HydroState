from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from hydrostate.data.site.training_smoke import representative_indices, verify_checkpoint_parameters
from hydrostate.experiments.gpu_memory import CudaMemoryMeter, has_memory_headroom


def test_representatives_include_each_actual_native_and_scalar_mask_maximum():
    product = SimpleNamespace(
        tasks={"et_ice": dict(support="native", grid="native"),
               "et_land": dict(support="native", grid="native"),
               "soil_moisture": dict(support="tile_mean_weak", grid="scalar")},
        table=pd.DataFrame(dict(sampling_group=["0"] * 4,
                               valid_label_counts=[{}, {"et_land_native_valid": 9},
                                                   {"et_ice_native_valid": 5},
                                                   {"soil_moisture_valid": 1}])))
    sites = [SimpleNamespace(table=pd.DataFrame(dict(valid_sm=[1], et_support=["static_210m"])))]
    dataset = SimpleNamespace(product=product, sites=sites, offsets=[0, 4])
    chosen = representative_indices(dataset)
    assert set(chosen) == {0, 1, 2, 3, 4}


def test_memory_retains_peaks_across_logger_reset_and_excludes_cache(monkeypatch):
    state = dict(allocated=30, reserved=43, external=1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda: ((48 - state["reserved"] - state["external"]) * 2**30, 48 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: state["reserved"] * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: state["reserved"] * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: state["allocated"] * 2**30)
    meter = CudaMemoryMeter()
    assert has_memory_headroom(meter.sample())
    state.update(allocated=10, reserved=30, external=2)
    report = meter.sample()
    assert report["peak_allocated_gib"] == 30
    assert report["peak_reserved_gib"] == 43
    assert report["peak_external_gib"] == 2
    assert report["memory_samples"] == 2
    assert has_memory_headroom(report)
    state.update(allocated=42, reserved=44, external=2)
    assert not has_memory_headroom(meter.sample())


@pytest.mark.parametrize("key,value", [("peak_allocated_gib", float("nan")),
                                      ("peak_external_gib", -1), ("gpu_total_gib", 0),
                                      ("memory_samples", 0)])
def test_invalid_memory_evidence_rejected(key, value):
    report = dict(peak_allocated_gib=30, peak_external_gib=1, gpu_total_gib=48, memory_samples=1)
    report[key] = value
    with pytest.raises(ValueError, match="memory evidence"):
        has_memory_headroom(report)


@pytest.mark.parametrize("reloads", [True, False])
def test_checkpoint_reload_requires_exact_restoration(tmp_path, reloads):
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "epoch_1.pth"
    torch.save(model.state_dict(), checkpoint)

    def load(path, strict):
        assert strict
        assert all(parameter.count_nonzero() == 0 for parameter in model.parameters())
        if reloads:
            model.load_state_dict(torch.load(path, weights_only=True), strict=strict)

    runner = SimpleNamespace(model=model, load_checkpoint=load)
    if reloads:
        assert verify_checkpoint_parameters(runner, checkpoint) == 2
    else:
        with pytest.raises(ValueError, match="did not restore"):
            verify_checkpoint_parameters(runner, checkpoint)
