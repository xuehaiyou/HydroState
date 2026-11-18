"""Scientific routing invariants and official complete MLoRE forward equivalence."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hydrostate.models.cosa.ple import ProgressiveExtraction
from hydrostate.models.mlore.decoder import FourDepthMLoRE, load_balance_loss


@pytest.mark.parametrize('widths', [(8,), (8, 4)])
def test_ple_private_isolation_and_progressive_shared_gradient(widths):
    model = ProgressiveExtraction(('sm', 'et'), 8, widths, tower_channels=4)
    value = torch.rand(2, 8, 3, 3, requires_grad=True)
    valid = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    valid[0] = False
    outputs = model(value, valid)
    assert not outputs['sm'][0].any()
    outputs['sm'].sum().backward()
    other_private = model.levels[0].private['et'][0][0].weight.grad
    if len(widths) == 1:
        assert other_private is None
    else:
        assert other_private is not None and other_private.abs().sum() > 0
        assert model.levels[0].shared_gate.weight.grad.abs().sum() > 0
        assert model.levels[-1].private['et'][0][0].weight.grad is None
    assert model.levels[-1].shared_gate is None
    assert not value.grad[0].any()


def test_ple_shared_gate_mixes_all_experts_and_task_gates_exclude_other_private():
    model = ProgressiveExtraction(('sm', 'et'), 1, (1, 1), tower_channels=1)
    level = model.levels[0]
    for gate in [*level.gates.values(), level.shared_gate]:
        nn.init.zeros_(gate.weight)
        nn.init.zeros_(gate.bias)
    for expert, constant in [(level.private['sm'][0], 2.),
                             (level.private['et'][0], 6.), (level.shared[0], 10.)]:
        nn.init.zeros_(expert[0].weight)
        nn.init.constant_(expert[0].bias, constant)
    value = torch.ones(1, 1, 2, 2)
    tasks, shared = level(dict(sm=value, et=value), value, value.bool())
    torch.testing.assert_close(tasks['sm'], value * 6)
    torch.testing.assert_close(tasks['et'], value * 8)
    torch.testing.assert_close(shared, value * 6)


class PatchTokens(nn.Module):
    def forward(self, value):
        return value.flatten(2).transpose(1, 2)


class IdentityTransformer(nn.Module):
    def forward(self, value):
        return value, None


def test_complete_mlore_matches_pinned_official_forward_and_balance():
    path = Path(__file__).parent / 'fixtures/mlore_official_reference.py'
    spec = importlib.util.spec_from_file_location('official_mlore', path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    torch.manual_seed(5)
    network = FourDepthMLoRE(('sm', 'et'), 8, 8, 3, (2, 4, 6), 2, 2, False).eval()
    reference = reference_module.OfficialDecoderReference()
    for name in ('norm', 'fea_fuse', 'MLoRE_1', 'MLoRE_2', 'task_mask'):
        setattr(reference, name, copy.deepcopy(getattr(network, name)))
    reference.patch_embed = PatchTokens()
    reference.pos_drop = nn.Identity()
    reference.pos_embed = nn.Parameter(torch.zeros(1, 10, 8))
    reference.blocks = nn.ModuleList([IdentityTransformer() for _ in range(4)])
    reference.select_list = [1, 2, 3]
    reference.num_layers = 4
    reference.resolution = (3, 3)
    reference.p = SimpleNamespace(TASKS=SimpleNamespace(NAMES=('sm', 'et')))
    reference.eval()
    value = torch.randn(2, 8, 3, 3, requires_grad=True)
    valid = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    actual, routes = network([value] * 4, valid)
    expected, info = reference(value)
    for task in actual:
        # Token-major versus channel-major layouts can change FP32 convolution rounding.
        torch.testing.assert_close(actual[task], expected[task], rtol=1e-6, atol=1e-7)
    official_loss = 0
    for layer in info['route_1_prob'] + info['route_2_prob']:
        weights = torch.cat(list(layer.values()), dim=0).mean(0).reshape(-1)
        official_loss += (weights.std() / weights.mean()) ** 2
    torch.testing.assert_close(load_balance_loss(routes, torch.ones(2, dtype=torch.bool)),
                               official_loss)
    (actual['sm'].square().mean() + load_balance_loss(routes, valid.flatten(1).any(1))).backward()
    assert value.grad.abs().sum() > 0
    assert network.MLoRE_1[0].share_conv.weight.grad.abs().sum() > 0
    assert network.MLoRE_2[3].conv2['sm'].W.weight.grad.abs().sum() > 0
    zero, _ = network([value.detach()] * 4, torch.zeros_like(valid))
    assert all(not x.any() for x in zero.values())
