import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.models.cosa.d2moe import _ROUTE, MixtureProjection, SemanticPriorInjector
from hydrostate.models.cosa.ditask import SpectralProjection


@pytest.mark.parametrize("task,cells", [(None, 64), ("sm", 4)])
def test_spectral_projection_matches_difw_forward_and_backward(task, cells):
    difw = pytest.importorskip("difw")
    torch.manual_seed(23)
    base = nn.Linear(14, 10)
    layer = SpectralProjection(base, ("sm",)).eval()
    x = torch.randn(2, 6, 14, requires_grad=True)
    theta = layer.shared if task is None else layer.private[task]
    reference_theta = theta.detach().clone().requires_grad_()
    # Official transform_data: torch.svd, warp the ordered samples, then U diag(s') V.T.
    u, singular, v = torch.svd(base.weight.detach())
    flow = difw.Cpab(cells, "pytorch", "cpu", zero_boundary=True)
    warped = flow.transform_data(singular.view(1, -1, 1), reference_theta, singular.numel())
    expected = F.linear(x, u @ warped.flatten().diag() @ v.T, base.bias)
    actual = layer(x, task)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    for grad, reference in zip(torch.autograd.grad(actual.square().sum(), (x, theta)),
                               torch.autograd.grad(expected.square().sum(), (x, reference_theta))):
        torch.testing.assert_close(grad, reference, atol=2e-4, rtol=5e-4)


def test_sparse_attention_projection_matches_active_experts_only():
    torch.manual_seed(14)
    proj = MixtureProjection(nn.Linear(7, 9), experts=16).eval()
    nn.init.normal_(proj.up)
    nn.init.normal_(proj.shared.b.weight)
    x = torch.randn(2, 5, 7)
    indices = torch.tensor([list(range(8)), list(range(3, 11))])
    probabilities = torch.randn(2, 8).softmax(1).requires_grad_()
    ctx = _ROUTE.set((indices, probabilities))
    try:
        actual = proj(x)
    finally:
        _ROUTE.reset(ctx)
    expected = proj.base(x).clone()
    for i in range(2):
        for weight, j in zip(probabilities[i], indices[i]):
            expected[i] += weight * F.linear(F.linear(x[i], proj.down[j]), proj.up[j])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert proj.up.grad[:11].abs().sum() > 0 and proj.up.grad[11:].count_nonzero() == 0
    assert probabilities.grad.abs().sum() > 0
    assert proj.shared.a.weight.grad is None
    torch.testing.assert_close(proj(x), proj.base(x) + proj.shared.b(proj.shared.a(x)) / 4)


def test_spi_uses_every_depth_and_masks_invalid_observations():
    spi = SemanticPriorInjector(4, 4, ["sm"], width=8).eval()
    valid = torch.ones(2, 1, 3, 5, dtype=torch.bool)
    valid[:, :, -1] = False
    levels = torch.randn(4, 2, 4, 3, 5, requires_grad=True)
    priors = torch.randn_like(levels, requires_grad=True)
    result = spi("sm", levels, spi.merge_prior(priors, valid), valid)
    assert result[:, :, -1].count_nonzero() == 0
    result.sum().backward()
    for tensor in (levels, priors):
        assert (tensor.grad.flatten(1).abs().sum(1) > 0).all()
        assert tensor.grad[..., -1, :].count_nonzero() == 0
