import torch

from hydrostate.models.cosa.trusted_bdta import readout_scores, relative_weights


def test_readout_scores_match_explicit_gradient_dots_and_preserve_training_graph():
    torch.manual_seed(8)
    readout = torch.nn.Conv2d(2, 3, 1)
    states = torch.randn(2, 2, 2, 2, requires_grad=True)
    predictions = readout(states).softmax(1)
    errors = predictions[0].flatten().square()
    trusted = (1 - predictions[1, 0]).square().mean()
    params = tuple(readout.parameters())
    gradient = torch.autograd.grad(trusted, params, retain_graph=True)
    expected = []
    for e in errors:
        row = torch.autograd.grad(e, params, retain_graph=True)
        expected.append(sum((a * b).sum() for a, b in zip(row, gradient)))
    scores = readout_scores(errors, trusted, params)
    torch.testing.assert_close(scores, torch.stack(expected))
    quality = torch.linspace(0.1, 0.9, len(scores))
    weights = relative_weights(scores, quality)
    assert not weights.requires_grad
    torch.testing.assert_close(weights.sum(), quality.sum())
    assert (weights / quality).argmax() == scores.argmax()
    torch.testing.assert_close(relative_weights(torch.ones_like(scores), quality), quality)
    (trusted + (errors * weights).mean()).backward()
    assert states.grad.isfinite().all() and states.grad.abs().sum() > 0
