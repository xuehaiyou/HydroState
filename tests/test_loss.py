import torch

from hydrostate.losses import MaskedHuberLoss


def test_masked_huber_ignores_invalid_values():
    prediction = torch.tensor([[[[1.0, 100.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[[[True, False]]]])
    loss = MaskedHuberLoss(delta=1.0)(prediction, target, valid)
    assert torch.isclose(loss, torch.tensor(0.5))
    loss.backward()
    assert prediction.grad[0, 0, 0, 1] == 0


def test_empty_mask_returns_differentiable_zero():
    prediction = torch.ones((2, 1, 4, 4), requires_grad=True)
    loss = MaskedHuberLoss()(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction))
    assert loss.item() == 0
    loss.backward()
    assert prediction.grad is not None

