import torch

from hydrostate.experiments.auto_lambda import virtual_sgd_step


def test_virtual_weights_match_finite_difference_of_station_error():
    theta = torch.tensor(0.4, requires_grad=True)
    weights = torch.tensor([0.7, 1.1], requires_grad=True)
    updated = virtual_sgd_step({'theta': theta}, (theta - 1).square(),
                               [(theta + 2).square(), None], weights, 0.1)
    meta = (updated['theta'] - 1.5).square()
    measured, = torch.autograd.grad(meta, weights)
    def error(weight):
        with torch.no_grad():
            virtual = theta - 0.1 * (2 * (theta - 1) + weight * 2 * (theta + 2))
            return (virtual - 1.5).square()
    eps = 1e-3
    finite = (error(weights[0] + eps) - error(weights[0] - eps)) / (2 * eps)
    torch.testing.assert_close(measured[0], finite, atol=1e-4, rtol=2e-4)
    assert measured[1] == 0 and theta.grad is None
