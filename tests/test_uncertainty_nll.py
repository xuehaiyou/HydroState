import copy
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from hydrostate.losses import UncertaintyWeightedNLL


def test_gaussian_nll_matches_reference_and_masks_nan(target_statistics):
    module = UncertaintyWeightedNLL(target_statistics)
    module.log_variances["water"].data.fill_(math.log(2.2))
    prediction = torch.tensor([0.1, 0.8, float("nan")], requires_grad=True)
    target = torch.tensor([0.2, 0.3, float("nan")])
    objective, count = module("water", prediction, target, torch.tensor([True, True, False]))
    expected = torch.nn.functional.gaussian_nll_loss(
        (prediction[:2] - 0.3) / 0.4,
        (target[:2] - 0.3) / 0.4,
        torch.full((2,), 2.2),
        full=True,
    )
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert count == 2 and prediction.grad[-1] == 0
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("residual,sign", [(2.0, -1), (0.0, 1)])
def test_variance_gradient_adapts_to_standardized_residual(target_statistics, residual, sign):
    module = UncertaintyWeightedNLL(target_statistics)
    prediction = torch.tensor([0.2 + 0.1 * residual], requires_grad=True)
    objective, _ = module("soil_moisture", prediction, torch.tensor([0.2]), torch.tensor([True]))
    objective.backward()
    grad = module.log_variances["soil_moisture"].grad
    torch.testing.assert_close(grad, torch.tensor(0.5 * (1 - residual**2)))
    assert grad.sign() == sign


def test_empty_task_has_no_error_or_variance_penalty(target_statistics):
    module = UncertaintyWeightedNLL(target_statistics)
    module.log_variances["et"].data.fill_(2.0)
    prediction = torch.full((3,), float("nan"), requires_grad=True)
    objective, count = module(
        "et", prediction, prediction.detach(), torch.zeros(3, dtype=torch.bool)
    )
    objective.backward()
    assert objective == 0 and count == 0
    assert module.log_variances["et"].grad == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_checkpoint_preserves_statistics_and_uncertainties(target_statistics, tmp_path):
    original = UncertaintyWeightedNLL(target_statistics)
    original.log_variances["et"].data.fill_(-0.7)
    path = tmp_path / "loss.pth"
    torch.save(original.state_dict(), path)
    restored = UncertaintyWeightedNLL(target_statistics)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert all(not b.requires_grad for b in restored.buffers())
    incompatible = copy.deepcopy(target_statistics)
    incompatible["tasks"]["et"]["std"] *= 2
    with pytest.raises(ValueError, match="normalization disagrees"):
        UncertaintyWeightedNLL(incompatible).load_state_dict(original.state_dict())


class _DistributedProbe(torch.nn.Module):
    def __init__(self, statistics):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(0.5))
        self.loss_module = UncertaintyWeightedNLL(statistics)

    def forward(self, x, y, valid):
        return self.loss_module("water", self.slope * x, y, valid)


def _ddp_check(rank, init_file, statistics):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        probe = DistributedDataParallel(_DistributedProbe(statistics), find_unused_parameters=True)
        for lengths in ((2, 3), (2, 0), (0, 0)):
            x = torch.arange(1, 1 + lengths[rank], dtype=torch.float32)
            y = torch.zeros_like(x)
            probe.zero_grad(set_to_none=True)
            objective, count = probe(x, y, torch.ones_like(x, dtype=torch.bool))
            objective.backward()
            # Independent concatenated reference, with no distributed loss helper.
            whole = torch.cat([torch.arange(1, 1 + n, dtype=torch.float32) for n in lengths])
            slope, log_var = (
                torch.tensor(0.5, requires_grad=True),
                torch.tensor(0.0, requires_grad=True),
            )
            nll_sum = torch.nn.functional.gaussian_nll_loss(
                (slope * whole - 0.3) / 0.4,
                torch.full_like(whole, -0.3 / 0.4),
                log_var.exp().expand_as(whole),
                full=True,
                reduction="sum",
            )
            expected = nll_sum / max(sum(lengths), 1)
            expected.backward()
            torch.testing.assert_close(objective, expected)
            assert count == sum(lengths)
            torch.testing.assert_close(probe.module.slope.grad, slope.grad)
            torch.testing.assert_close(
                probe.module.loss_module.log_variances["water"].grad, log_var.grad
            )
    finally:
        dist.destroy_process_group()


def test_ddp_global_average_matches_concatenated_gradient(target_statistics, tmp_path):
    mp.spawn(
        _ddp_check, args=(str(tmp_path / "rendezvous"), target_statistics), nprocs=2, join=True
    )
