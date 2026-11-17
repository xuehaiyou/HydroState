import torch
from mmengine.config import Config

from hydrostate.models.cosa.cgc import CGCRouter


def test_cgc_private_gradient_isolation_and_missing_observations():
    router = CGCRouter(('sm', 'et'), state_channels=4, channels=3)
    state = torch.randn(2, 4, 3, 3, requires_grad=True)
    valid = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    valid[0] = False
    valid[1, :, 0, 0] = False
    features = {t: torch.randn(2, 3, 3, 3, requires_grad=True) for t in ('sm', 'et')}
    out = router(state, features, valid)
    assert torch.isfinite(out['sm']).all()
    assert not out['sm'][0].any()
    assert not out['sm'][1, :, 0, 0].any()
    out['sm'].square().sum().backward()
    assert features['sm'].grad.abs().sum() > 0
    assert features['et'].grad is None
    assert router.gates['et'].weight.grad is None
    assert all(e.project.weight.grad.abs().sum() > 0 for e in router.shared)
    assert not state.grad[0].any()


def test_cgc_zero_logits_mix_only_shared_and_own_private():
    router = CGCRouter(('sm', 'et'), state_channels=4, channels=3)
    for gate in router.gates.values():
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
    state = torch.randn(1, 4, 2, 2)
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    features = dict(sm=torch.ones(1, 3, 2, 2), et=torch.full((1, 3, 2, 2), 100.))
    expected = (sum(e(state, valid) for e in router.shared) + features['sm']) / 3
    torch.testing.assert_close(router(state, features, valid)['sm'], expected)


def test_cgc_configuration_changes_only_router():
    base = Config.fromfile('configs/hydrostate_joint.py')
    cgc = Config.fromfile('configs/hydrostate_joint_cgc.py')
    cgc.model.router = base.model.router
    cgc.work_dir = base.work_dir
    assert cgc.to_dict() == base.to_dict()
