import torch
from mmengine.config import Config
from test_multifm_rutal import sequence

from hydrostate.models.cosa.bdta import BDTAMoTEDecoder
from hydrostate.models.cosa.multifm_bdta import MultiFMBDTADecoder


def test_multifm_bdta_protocol_changes_only_source_weighting():
    cfg = Config.fromfile("configs/hydrostate_joint_multifm_bdta.py")
    equal = Config.fromfile("configs/hydrostate_joint_multifm_bdta_equal.py")
    assert cfg.model.encoder.method == "bdta" and cfg.model.anysat.frozen
    assert cfg.model.anysat.ladder_features and cfg.model.encoder.out_indices == [2, 5, 8, 11]
    assert cfg.model.decoder.learn_source_weights and not equal.model.decoder.learn_source_weights
    equal.model.decoder.learn_source_weights = True
    assert cfg.to_dict() == equal.to_dict()


def test_masked_source_reduces_to_bdta_and_both_missing_are_finite():
    torch.manual_seed(13)
    opts = dict(channels=24, in_channels=24, num_levels=5, num_heads=2, bypass_heads=2)
    original, dual = BDTAMoTEDecoder(**opts).eval(), MultiFMBDTADecoder(**opts).eval()
    missing, unexpected = dual.load_state_dict(original.state_dict(), strict=False)
    assert not unexpected and all(k == "source_scale" or k.startswith("anysat_projections.")
                                  for k in missing)
    dual.source_scale.data.fill_(2)
    valid = torch.rand(2, 3, 4, 5) > .35
    valid[0] = False
    target = torch.tensor([[1600000000, 1600000800], [1600000300, 1600023000]])
    groups = ("coverage", "sm", "et")
    one = dict(shared=sequence(valid))
    both = dict(**one, anysat=sequence(torch.zeros_like(valid)))
    a, ma = original.states(one, target, groups)
    b, mb = dual.states(both, target, groups)
    torch.testing.assert_close(ma, mb)
    for task in groups:
        torch.testing.assert_close(a[task], b[task], rtol=2e-5, atol=2e-5)
        assert (b[task][0] == 0).all()


def test_learned_starts_as_equal_then_selects_by_task_and_learns():
    opts = dict(channels=24, in_channels=24, num_levels=5, num_heads=2, bypass_heads=2)
    learned = MultiFMBDTADecoder(**opts).eval()
    equal = MultiFMBDTADecoder(learn_source_weights=False, **opts).eval()
    equal.load_state_dict(learned.state_dict(), strict=False)
    keys = torch.tensor([[1., 0.], [-1., 0.]]).reshape(2, 1, 1, 1, 2)
    queries = {"sm": keys[0], "et": keys[1]}
    active = torch.ones(2, 1, 1, 1, dtype=torch.bool)
    learned.source_scale.data.fill_(2)
    out = learned.source_context(queries, keys, active)
    assert out["sm"][..., 0] > 0.9 and out["et"][..., 0] < -0.9
    learned.source_scale.data.zero_()
    torch.manual_seed(42)
    mask = torch.ones(1, 2, 3, 4, dtype=torch.bool)
    data = dict(shared=sequence(mask), anysat=sequence(mask))
    target = torch.tensor([[1600000000, 1600086400]])
    states, _ = learned.states(data, target, ("sm", "et"))
    fixed, _ = equal.states(data, target, ("sm", "et"))
    for task in states:
        torch.testing.assert_close(states[task], fixed[task])
    sum(v.square().mean() for v in states.values()).backward()
    assert learned.source_scale.grad.abs() > 0
    for seq in data.values():
        assert all(x.grad is not None and x.grad.abs().sum() > 0 for x, _, _ in seq[0])
