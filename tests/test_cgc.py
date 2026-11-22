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


def test_private_sm_bypasses_shared_experts():
    router = CGCRouter(('sm', 'et'), 4, 3, private_only=('sm',))
    state = torch.randn(1, 4, 3, 3)
    valid = torch.ones_like(state[:, :1], dtype=torch.bool)
    valid[:, :, 1, 2] = False
    features = {task: torch.randn(1, 3, 3, 3, requires_grad=True) for task in router.experts}
    values = router(state, features, valid)
    torch.testing.assert_close(features['sm'].masked_fill(~valid, 0), values['sm'])
    values['sm'].sum().backward()
    assert set(router.gates) == {'et'}
    assert features['sm'].grad is not None
    assert all(p.grad is None for p in router.shared.parameters())


def test_spatial_sm_gate_changes_only_sm_and_matches_direct_mixture():
    base = CGCRouter(('sm', 'et'), 4, 3)
    local = CGCRouter(('sm', 'et'), 4, 3, spatial_gates=('sm',))
    local.load_state_dict(base.state_dict(), strict=True)
    state = torch.randn(2, 4, 3, 3)
    valid = torch.ones_like(state[:, :1], dtype=torch.bool)
    valid[:, :, 0, 1] = False
    features = {task: torch.randn(2, 3, 3, 3) for task in base.experts}
    values = local(state, features, valid)
    torch.testing.assert_close(values['et'], base(state, features, valid)['et'])
    assert not values['sm'][:, :, 0, 1].any()
    experts = [expert(state.masked_fill(~valid, 0), valid) for expert in local.shared]
    weights = local.gates['sm'](state[:, :, 1, 2]).softmax(-1)
    expected = sum(feature[:, :, 1, 2] * weights[:, i:i + 1]
                   for i, feature in enumerate([*experts, features['sm']]))
    torch.testing.assert_close(values['sm'][:, :, 1, 2], expected)


def test_tied_refinement_starts_at_local_cgc_and_keeps_private_paths_isolated():
    from hydrostate.models.cosa.cgc import RecurrentCGCRouter

    groups = ('sm', 'et')
    torch.manual_seed(42)
    ref = CGCRouter(groups, 4, 3, spatial_gates=['sm'])
    torch.manual_seed(42)
    model = RecurrentCGCRouter(groups, 4, 3, depth=3)
    x = torch.randn(2, 4, 3, 3)
    valid = torch.ones_like(x[:, :1], dtype=torch.bool)
    valid[0] = False
    features = {t: torch.randn(2, 3, 3, 3, requires_grad=True) for t in groups}
    result = model(x, features, valid)
    for task, value in ref(x, features, valid).items():
        torch.testing.assert_close(result[task], value, rtol=0, atol=0)
    result['sm'].sum().backward()
    assert model.step.grad.abs() > 0
    with torch.no_grad():
        model.step.fill_(.5)
    model.zero_grad(set_to_none=True)
    calls = []
    hook = model.refinement.shared[0].register_forward_hook(lambda *args: calls.append(1))
    result = model(x, features, valid)
    hook.remove()
    assert len(calls) == 2 and not result['sm'][0].any()
    result['sm'].sum().backward()
    assert features['et'].grad is None
    assert model.private['et'].project.weight.grad is None
    assert model.private['sm'].project.weight.grad.abs().sum() > 0
    assert model.refinement.shared[0].project.weight.grad.abs().sum() > 0


def test_learned_locality_recovers_both_gate_scales_and_shares_three_scalars():
    from hydrostate.models.cosa.cgc import RecurrentCGCRouter

    groups = ('coverage', 'sm', 'et')
    model = RecurrentCGCRouter(groups, 4, 3, depth=2, spatial_gates=(), learn_locality=True)
    assert len(model.locality) == 3
    state = torch.randn(2, 4, 3, 3)
    valid = torch.ones_like(state[:, :1], dtype=torch.bool)
    valid[0] = False
    valid[1, :, 0, 0] = False
    features = {t: torch.randn(2, 3, 3, 3) for t in groups}
    with torch.no_grad():
        model.step.fill_(.4)
    outputs = model(state, features, valid)
    sum(y.square().sum() for y in outputs.values()).backward()
    for name in groups:
        assert model.locality[name].grad.abs() > 0
        assert not outputs[name][0].any()
    for logit, spatial in ((-100., ()), (100., groups)):
        ref = RecurrentCGCRouter(groups, 4, 3, depth=2, spatial_gates=spatial)
        ref.load_state_dict({k: v for k, v in model.state_dict().items()
                             if not k.startswith('locality.')})
        with torch.no_grad():
            for p in model.locality.values():
                p.fill_(logit)
        for task, value in ref(state, features, valid).items():
            torch.testing.assert_close(model(state, features, valid)[task], value)


def test_factorized_gate_recovers_endpoints_and_mixed_marginals():
    from hydrostate.models.cosa.cgc import split_gate_weights

    tile = torch.randn(2, 1, 1, 3)
    local = torch.randn(2, 3, 4, 3)
    for scores in (tile, local):
        torch.testing.assert_close(split_gate_weights(scores, scores), scores.softmax(-1))
    for shared, private in ((tile, local), (local, tile)):
        weights = split_gate_weights(shared, private)
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
        torch.testing.assert_close(weights[..., -1], private.softmax(-1)[..., -1].expand(2, 3, 4))
        relative = weights[..., :2] / weights[..., :2].sum(-1, keepdim=True)
        torch.testing.assert_close(relative, shared[..., :2].softmax(-1).expand_as(relative))
