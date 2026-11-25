import pytest
import torch
from mmengine.config import Config
from test_multifm_rutal import sequence

from hydrostate.models.cosa.bdta import BDTAMoTEDecoder
from hydrostate.models.cosa.direct_bdta import DirectBDTADecoder, DirectSourceAttention

OPTIONS = dict(channels=24, in_channels=24, num_levels=5, num_heads=2, bypass_heads=2)
TASKS = ("coverage", "sm", "et")
BOUNDS = torch.tensor([[1600000000, 1600086400], [1600000300, 1600023000]])


def test_direct_and_equal_protocol():
    learned = Config.fromfile("configs/hydrostate_joint_direct_bdta.py")
    equal = Config.fromfile("configs/hydrostate_joint_direct_bdta_equal.py")
    assert learned.model.type == "DirectMultiFMBDTA"
    assert learned.model.decoder.type == "DirectBDTADecoder"
    assert learned.model.anysat.frozen and learned.model.encoder.method == "bdta"
    assert (
        learned.model.decoder.learn_source_weights and not equal.model.decoder.learn_source_weights
    )
    equal.model.decoder.learn_source_weights = True
    assert learned.to_dict() == equal.to_dict()


def test_source_marginal_and_spatial_weights():
    logits = torch.tensor(
        [[[1.0, 3.0], [2.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [4.0, 2.0], [1.0, 2.0]]],
        requires_grad=True,
    )
    support = torch.tensor([[True, False, False], [True, True, False]])
    learned = DirectSourceAttention.point_weights(logits, support, True)
    fixed = DirectSourceAttention.point_weights(logits, support, False)
    torch.testing.assert_close(learned[:, 0], logits[:, 0].reshape(-1).softmax(0).reshape(2, 2))
    torch.testing.assert_close(fixed.sum(-1), torch.tensor([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]]))
    torch.testing.assert_close(fixed[:, 0] * 2, logits[:, 0].softmax(-1))
    torch.testing.assert_close(learned[:, 1:], fixed[:, 1:])
    (fixed[:, 0, 1].sum() + learned[0, 0].sum()).backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad[:, 0].abs() > 0).all()


def run_stages(model, data):
    stages = []
    handles = [
        stage.register_forward_hook(lambda m, args, out: stages.append(out))
        for stage in model.stages
    ]
    try:
        heads, mask = model.states(data, BOUNDS, TASKS)
        return [*stages, heads], mask
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("second_source", [False, True])
def test_single_fm_matches_four_stages_and_readouts(second_source):
    torch.manual_seed(3407)
    base = BDTAMoTEDecoder(**OPTIONS).eval()
    direct = DirectBDTADecoder(**OPTIONS).eval()
    missing, unexpected = direct.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(
        "extra_" in k or k.startswith("anysat_projections") for k in missing
    )
    valid = torch.rand(2, 3, 5, 7) > 0.5
    valid[0] = False
    valid[:, :, 2, 3] = False
    data = dict(shared=sequence(valid))
    ref, mask = run_stages(base, data)
    if second_source:
        data["anysat"] = sequence(torch.zeros_like(valid))
    actual, actual_mask = run_stages(direct, data)
    torch.testing.assert_close(actual_mask, mask)
    assert len(ref) == len(actual) == 5
    for stage, expected in zip(actual, ref, strict=True):
        for task in TASKS:
            torch.testing.assert_close(stage[task], expected[task], atol=2e-5, rtol=2e-5)
            assert (stage[task][0] == 0).all()


@pytest.mark.parametrize("learned", [True, False])
def test_direct_paths_train_both_sources_and_reverse_attention(learned):
    torch.manual_seed(100)
    model = DirectBDTADecoder(learn_source_weights=learned, **OPTIONS).eval()
    valid = torch.ones(2, 3, 5, 7, dtype=torch.bool)
    data = dict(shared=sequence(valid), anysat=sequence(valid))
    result, _ = model.states(data, BOUNDS, TASKS)
    sum(x.square().mean() for x in result.values()).backward()
    for seq in data.values():
        for feature, _, _ in seq[0]:
            assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    for stage in model.stages:
        attn = stage.g2t.attention
        for param in (
            attn.offset.weight,
            attn.extra_offsets[0].weight,
            attn.weight.weight,
            attn.extra_weights[0].weight,
            stage.t2g.attention.value.weight,
        ):
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
