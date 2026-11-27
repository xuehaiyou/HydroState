from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from hydrostate.datasets.full_source_sampler import FullSourceSampler
from hydrostate.losses.fixed_mse import FixedStandardizedMSE
from hydrostate.tasks import LEGACY_TASKS


def test_quality_masked_mse_is_a_pure_squared_error(target_statistics):
    loss = FixedStandardizedMSE(target_statistics, LEGACY_TASKS)
    x = torch.tensor([0.4, 0.8, torch.nan], requires_grad=True)
    y = torch.tensor([0.4, 0.4, torch.nan])
    value = loss.local_sum(
        "water", x, y, torch.tensor([True, True, False]), weight=torch.tensor([1.0, 0.5, torch.nan])
    )
    torch.testing.assert_close(value, torch.tensor(0.5))
    value.backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 2.5, 0.0]))
    assert not list(loss.parameters())


def test_full_sampler_unique_including_two_window_tail():
    class Sources:
        product = range(24)
        sites = [
            SimpleNamespace(
                table=pd.DataFrame(dict(valid_sm=[1] * 37 + [0] * 3, valid_et=[0] * 37 + [1] * 3))
            )
        ]
        offsets = [0, 24, 64]
        base_length = 64

        def __len__(self):
            return 82

    sampler = FullSourceSampler(Sources())
    result = list(sampler)
    assert sorted(result) == list(range(82))
    chunks = [np.array(result[i : i + 10]) for i in range(0, len(result), 10)]
    assert all((c < 24).any() and ((24 <= c) & (c < 61)).any() for c in chunks)
    assert all((c >= 64).any() for c in chunks[:-1])
    assert sum(((61 <= c) & (c < 64)).any() for c in chunks) == 3
    sampler.set_epoch(1)
    assert list(sampler) != result


def test_equal_product_groups_and_missing_output(monkeypatch):
    from hydrostate.models.cosa.fixed_bdta import FixedSupervisionBDTA
    from hydrostate.models.cosa.foundation_v2 import FoundationStageModelV2

    def terms(*args):
        return dict(
            loss_water=torch.tensor(3.0),
            valid_count_water=torch.tensor(2),
            loss_ice=torch.tensor(0.0),
            valid_count_ice=torch.tensor(0),
            loss_land=torch.tensor(9.0),
            valid_count_land=torch.tensor(1),
            loss_sm=torch.tensor(0.0),
            valid_count_sm=torch.tensor(0),
            loss_site_sm=torch.tensor(0.25),
        )

    monkeypatch.setattr(FoundationStageModelV2, "_loss_from_outputs", terms)
    model = FixedSupervisionBDTA.__new__(FixedSupervisionBDTA)
    torch.nn.Module.__init__(model)
    model.expert_groups = {"cover": ("water", "ice", "land"), "sm": ("sm",)}
    result = model._loss_from_outputs({}, {})
    torch.testing.assert_close(result["loss_product_cover"], torch.tensor(6 / 9))
    assert result["loss_product_sm"] == 0 and result["loss_site_sm"] == 0.25
