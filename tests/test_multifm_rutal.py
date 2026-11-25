import pytest
import torch
from test_mtlora import tiny  # noqa: F401 -- shared tiny OlmoEarth fixture

from hydrostate.models.cosa.multifm_rutal import MultiFMRUTaLDecoder, source_weights
from hydrostate.models.cosa.rutal import RUTaLDecoder


def sequence(valid):
    b, t, h, w = valid.shape
    times = torch.arange(t).expand(b, t) * 86400 + 1600000000
    levels = [(torch.randn(b, t, h, w, 24, requires_grad=True), valid, times) for _ in range(5)]
    return levels, torch.zeros(b, 24, h, w), valid.any(1)[:, None]


def test_cosine_selects_a_different_source_per_point_and_respects_support():
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    opposite = -query
    keys = torch.stack(
        (
            torch.stack((query[0], opposite[1], query[2], query[3])),
            torch.stack((opposite[0], query[1], query[2], query[3])),
        )
    )
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    weights = source_weights(query, keys, valid, torch.tensor(3.0))
    assert weights[0, 0] > 0.99 and weights[1, 1] > 0.99
    torch.testing.assert_close(weights[:, 2:], torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    torch.testing.assert_close(
        source_weights(query, keys, valid, None)[:, :2], torch.full((2, 2), 0.5)
    )


@pytest.mark.parametrize("disabled_in_graph", [False, True])
def test_single_source_reproduces_all_four_stages_and_three_readouts(disabled_in_graph):
    torch.manual_seed(3407)
    opts = dict(channels=24, in_channels=24, num_levels=5, num_heads=2, ladder_heads=2)
    base, dual = RUTaLDecoder(**opts).eval(), MultiFMRUTaLDecoder(**opts).eval()
    dual.load_state_dict(base.state_dict(), strict=False)
    dual.source_scale.data.fill_(2.0)
    mask = torch.rand(2, 3, 5, 7) > 0.35
    mask[:, :, 2, 3] = False
    data = dict(shared=sequence(mask))
    target = torch.tensor([[1600000000, 1600000800], [1600000300, 1600023000]])
    tasks = ("coverage", "sm", "et")

    def capture(model, inputs):
        stages = []
        handles = [
            s.register_forward_hook(lambda mod, args, result: stages.append(result))
            for s in model.stages
        ]
        try:
            states, valid = model.states(inputs, target, tasks)
            return stages, states, valid
        finally:
            for handle in handles:
                handle.remove()

    s0, f0, m0 = capture(base, data)
    if disabled_in_graph:
        data["anysat"] = sequence(torch.zeros_like(mask))
    s1, f1, m1 = capture(dual, data)
    torch.testing.assert_close(m1, m0)
    assert len(s0) == len(s1) == 12
    for x, y in zip(s0, s1):
        # Stage.forward precedes masking the output, so compare only valid queries.
        valid = mask.any(1)[..., None].expand_as(x)
        torch.testing.assert_close(x[valid], y[valid], atol=2e-5, rtol=2e-5)
    for task in tasks:
        torch.testing.assert_close(f0[task], f1[task], atol=2e-5, rtol=2e-5)


def test_learned_starts_at_fixed_and_updates_both_sources_and_score():
    torch.manual_seed(42)
    opts = dict(channels=24, in_channels=24, num_levels=5, num_heads=2, ladder_heads=2)
    learned = MultiFMRUTaLDecoder(**opts)
    equal = MultiFMRUTaLDecoder(learn_source_weights=False, **opts)
    equal.load_state_dict(learned.state_dict(), strict=False)
    valid = torch.ones(1, 2, 4, 6, dtype=torch.bool)
    data = dict(shared=sequence(valid), anysat=sequence(valid))
    target = torch.tensor([[1600000000, 1600086400]])
    states, _ = learned.states(data, target, ("sm", "et"))
    fixed, _ = equal.states(data, target, ("sm", "et"))
    loss = sum(state.square().mean() for state in states.values())
    loss.backward()
    for key in states:
        torch.testing.assert_close(states[key], fixed[key])
    assert torch.isfinite(learned.source_scale.grad) and learned.source_scale.grad.abs() > 0
    for seq in data.values():
        for features, _, _ in seq[0]:
            assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0


@pytest.mark.usefixtures("tiny")
def test_all_seven_physical_outputs_match_original_with_anysat_masked(tmp_path, monkeypatch):
    from retrieval.test_model import batch
    from test_cosa import config

    from hydrostate.models.cosa.anysat import AnySatEncoder
    from hydrostate.registry import MODELS

    # Exercise the real model entry, time queries, task heads and physical constraints.
    def inactive_anysat(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.backbone = torch.nn.Identity()
        self.ladder_features = True

    monkeypatch.setattr(AnySatEncoder, "__init__", inactive_anysat)
    cfg = config(tmp_path, False)
    cfg.model.type = "FoundationStageModelV2"
    cfg.model.encoder.update(
        type="FoundationStageEncoderV2", method="rutal", num_levels=4, out_indices=[0, 1, 3, 5]
    )
    cfg.model.decoder.update(type="RUTaLDecoder", num_levels=5, ladder_heads=2)
    reference = MODELS.build(cfg.model).eval()
    cfg.model.type = "MultiFMRUTaL"
    cfg.model.decoder.type = "MultiFMRUTaLDecoder"
    cfg.model.anysat = dict(type="AnySatEncoder")
    cfg.model.sources = ("olmoearth",)
    dual = MODELS.build(cfg.model).eval()
    missing, unexpected = dual.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and all(
        k == "decoder.source_scale" or k.startswith("decoder.anysat_projections.") for k in missing
    )
    with torch.no_grad():
        samples = reference.data_preprocessor(batch(), training=False)
        a, b = (m(**samples, mode="tensor") for m in (reference, dual))
    assert a.keys() == b.keys() and len(a) == 7
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=2e-5, rtol=2e-5)
