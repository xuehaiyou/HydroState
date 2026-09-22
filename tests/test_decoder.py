import torch

from hydrostate.models import RegressionHead, SharedUNetDecoder


def test_decoder_and_head_shapes():
    decoder = SharedUNetDecoder(in_channels=16, channels=[16, 8, 4], scale_factor=4)
    head = RegressionHead(in_channels=4, activation="sigmoid")
    output = head(decoder(torch.randn(2, 16, 32, 32)))
    assert output.shape == (2, 1, 128, 128)
    assert torch.all((output >= 0) & (output <= 1))


def test_model_outputs_30m_grid():
    from hydrostate.models.hydrostate_model import HydroStateModel
    from hydrostate.registry import MODELS

    class TestEncoder(torch.nn.Module):
        def forward(self, inputs):
            return inputs

    MODELS.register_module(name="GridTestEncoder", module=TestEncoder, force=True)
    model = HydroStateModel(
        encoder=dict(type="GridTestEncoder"),
        decoder=dict(type="SharedUNetDecoder", in_channels=4, channels=[4, 4, 4]),
        heads={name: dict(type="RegressionHead", in_channels=4, activation="sigmoid")
               for name in ["water", "soil_moisture", "et"]},
        loss=dict(type="MaskedHuberLoss"),
    )
    outputs = model(torch.randn(1, 4, 30, 30))
    assert all(value.shape == (1, 1, 40, 40) for value in outputs.values())
    sum(value.mean() for value in outputs.values()).backward()


def test_broadcast_attention_matches_dense_mask():
    from types import SimpleNamespace
    from hydrostate.models.olmoearth_encoder import _broadcast_sdpa
    torch.manual_seed(4)
    q = torch.randn(2, 3, 7, 8, requires_grad=True)
    k = torch.randn(2, 3, 9, 8, requires_grad=True)
    v = torch.randn(2, 3, 9, 8, requires_grad=True)
    mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None].repeat(1, 3, 7, 1))
    actual = _broadcast_sdpa(SimpleNamespace(attn_drop=SimpleNamespace(p=0)),
                             q, k, v, n=7, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    dense_grads = torch.autograd.grad(expected.sum(), (q, k, v), retain_graph=True)
    sparse_grads = torch.autograd.grad(actual.sum(), (q, k, v))
    for left, right in zip(dense_grads, sparse_grads):
        torch.testing.assert_close(left, right)
