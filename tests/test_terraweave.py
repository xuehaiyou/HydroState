import pytest
import torch
from mmengine.config import Config
from test_mtlora import tiny  # noqa: F401 -- miniature OlmoEarth fixture
from test_multifm_rutal import sequence

from hydrostate.models.cosa.terraweave import (
    TerraWeaveDecoder,
    UnifiedReciprocalAttention,
    partition,
    unpartition,
)


def test_window_geometry_and_padding():
    marker = torch.arange(900).reshape(1, 1, 30, 30, 1)
    windows = partition(marker, 4)
    assert windows.shape == (64, 1, 16, 1) and windows[63, 0, 5] == 899
    assert partition(torch.ones_like(marker, dtype=torch.bool), 4)[63].sum() == 4
    torch.testing.assert_close(unpartition(windows, 30, 30, 4)[:, :, 0], marker[..., 0])


def test_reciprocal_formula_and_missing_keys():
    torch.manual_seed(19)
    block = UnifiedReciprocalAttention(8, 1)
    z, e = torch.randn(2, 6, 8), torch.randn(2, 4, 8)
    ev = torch.tensor([[True, False, True, True], [False] * 4])
    zv = torch.tensor([[True, True, False, True, False, True], [False] * 6])
    result = block(z, e, zv, ev)

    # Independently compute reverse and direct rows; A is identical on read and return.
    def weights(q, k, mask):
        logits = block.q(block.norm(q)) @ block.k(block.norm(k)).transpose(-1, -2)
        return (logits / 8**0.5).masked_fill(~mask[:, None], -torch.inf).softmax(-1)

    z, e = z[:1], e[:1]
    a = weights(z, e, ev[:1])
    u = z + block.out(a @ block.v(block.norm(e)))
    rev = weights(e, u, zv[:1]) @ block.v(block.norm(u))
    direct = weights(u, u, zv[:1]) @ block.v(block.norm(u))
    u = u + block.out(block.gamma_return * (a @ rev) + block.gamma_variable * direct)
    expected = (u + block.ffn(u)).masked_fill(~zv[:1, :, None], 0)
    torch.testing.assert_close(result[:1], expected)
    assert result[1].count_nonzero() == 0 and result.isfinite().all()


def test_four_independent_stages_masks_target_time_and_backward():
    torch.manual_seed(105)
    net = TerraWeaveDecoder(in_channels=24, channels=24, num_heads=2)
    valid = torch.ones(2, 2, 5, 7, dtype=torch.bool)
    valid[0] = False
    valid[:, :, 2, 1] = False
    data = {}
    for name in ("shared", "anysat"):
        levels, quality, mask = sequence(valid)
        data[name] = (levels[:4], quality, mask)
    bounds = torch.tensor([[1600000000, 1600086400], [1600000300, 1600023000]])
    calls = []

    def record_input(module, args):
        calls.append((module, args[0].detach()))

    handles = [stage.register_forward_pre_hook(record_input) for stage in net.stages]
    states, mask = net.states(data, bounds, net.groups)
    for handle in handles:
        handle.remove()
    assert len(net.stages) == 4 and [stage for stage, _ in calls] == list(net.stages)
    pointers = [p.data_ptr() for stage in net.stages for p in stage.parameters()]
    assert len(pointers) == len(set(pointers))
    # The initial state is independent of the image, even for all-missing windows.
    torch.testing.assert_close(calls[0][1][0], calls[0][1][-1])
    assert not torch.equal(calls[1][1][0], calls[1][1][-1])
    for state in states.values():
        assert state.shape == (2, 24, 5, 7)
        assert not state.masked_select(~mask.expand_as(state)).any()
    changed, _ = net.states(data, bounds + 60 * 86400, ("et",))
    assert not torch.allclose(changed["et"][1], states["et"][1])
    states["et"].square().mean().backward()
    for p in (
        *net.stages.parameters(),
        net.pooler.time_query[0].weight,
        net.source_id.weight,
        net.variable_id.weight,
        net.position.weight,
    ):
        assert p.grad.isfinite().all() and p.grad.abs().sum() > 0
    for levels, _, _ in data.values():
        for feature, _, _ in levels:
            assert feature.grad.isfinite().all() and feature.grad.abs().sum() > 0


@pytest.mark.usefixtures("tiny")
def test_real_olmo_entry_seven_outputs_and_frozen_encoder(tmp_path, monkeypatch):
    from retrieval.test_model import batch
    from test_cosa import config

    from hydrostate.models.cosa.anysat import AnySatEncoder
    from hydrostate.registry import MODELS

    def stub_anysat(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.frozen, self.ladder_features, self.include_patch = True, True, False
        self.backbone = torch.nn.Identity()

    monkeypatch.setattr(AnySatEncoder, "__init__", stub_anysat)
    cfg = config(tmp_path, False)
    variant = Config.fromfile("configs/hydrostate_joint_terraweave.py")
    cfg.model.type = variant.model.type
    cfg.model.encoder.update(
        type="FoundationStageEncoderV2", method="terraweave", out_indices=[0, 1, 3, 5], num_levels=4
    )
    cfg.model.decoder.update(type="TerraWeaveDecoder", num_levels=4, num_heads=2)
    cfg.model.anysat = dict(type="AnySatEncoder")
    cfg.model.sources = ("olmoearth",)
    model = MODELS.build(cfg.model).train()
    assert not model.encoder.backbone.training and not any(
        p.requires_grad for p in model.encoder.parameters()
    )
    data = model.data_preprocessor(batch(), training=True)
    result = model(**data, mode="tensor")
    assert set(result) == set(model.tasks) and len(result) == 7
    frac = sum(result[g] for g in ("water", "snow_ice", "land"))
    torch.testing.assert_close(frac, torch.ones_like(frac))
    assert all(result[t].min() >= 0 for t in ("et_water", "et_ice", "et_land"))
    sum(x.square().mean() for x in result.values()).backward()
    assert all(stage.gamma_return.grad.isfinite() for stage in model.decoder.stages)
