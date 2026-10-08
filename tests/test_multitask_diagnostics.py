import pandas as pd
import pytest
import torch

from hydrostate.losses import UncertaintyWeightedNLL
from tools.validation.diagnose_multitask import FIELDS, metrics


def test_sufficient_statistics_preserve_observation_weighting():
    targets = [torch.tensor([1., 3.]), torch.tensor([2.])]
    predictions = [torch.tensor([2., 2.]), torch.tensor([5.])]
    rows = []
    for y, p in zip(targets, predictions):
        e = p-y
        rows.append(dict(count=len(y), abs=e.abs().sum().item(), sq=e.square().sum().item(),
                         err=e.sum().item(), y=y.sum().item(), y2=y.square().sum().item()))
    actual = metrics(pd.DataFrame(rows))
    y, p = torch.cat(targets), torch.cat(predictions)
    assert actual['rmse'] == pytest.approx(((p-y).square().mean().sqrt()).item())
    assert actual['r2'] == pytest.approx((1-(p-y).square().sum()/(y-y.mean()).square().sum()).item())
    assert actual['bias'] == pytest.approx((p-y).mean().item())
    assert metrics(pd.DataFrame(columns=FIELDS)) == {'count': 0}


@pytest.mark.parametrize('task', ['water', 'soil_moisture', 'et'])
def test_nll_gradient_rescaling_matches_actual_loss(target_statistics, task):
    loss = UncertaintyWeightedNLL(target_statistics)
    loss.log_variances[task].data.fill_(-0.4)
    slope = torch.tensor(.7, requires_grad=True)
    x = torch.tensor([.3, .6, .9])
    y = torch.tensor([.1, .5, float('nan')])
    valid = torch.tensor([True, True, False])
    nll, _ = loss(task, slope*x, y, valid)
    actual = torch.autograd.grad(nll, slope)[0]
    std = target_statistics['tasks'][task]['std']
    mse = (((slope*x)[valid]-y[valid])/std).square().mean()
    expected = torch.autograd.grad(mse, slope)[0] / (2*loss.variance(task).detach())
    torch.testing.assert_close(actual, expected)
