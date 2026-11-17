"""Frozen baseline equivalence, expert masks and fusion-only gradients."""

import copy

import torch
from mmengine.config import Config
from test_mtlora import config, tiny  # noqa: F401 -- shared tiny FM fixture
from test_multilevel_temporal_model import raw_inputs

from hydrostate.models.adapterfusion.fusion import ResidualAdapterFusion
from hydrostate.registry import MODELS


def test_joint_experts_preserve_lora_policy_and_receive_gradients(tmp_path):
    cfg = config(tmp_path).model
    cfg.encoder.private_rank = 0
    baseline = MODELS.build(copy.deepcopy(cfg))
    policy = {n: p.requires_grad for n, p in baseline.named_parameters()}
    cfg.update(type='AdapterFusionHydroStateModel', baseline_checkpoint=None,
               freeze_baseline=False)
    model = MODELS.build(cfg).train()
    assert model.queries.training and model.decoders.training and model.fusions.training
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (True if name.startswith('fusions.') else policy[name])
    x = raw_inputs(model.tasks)
    x['target_time']['soil_moisture'] += 86400
    for fusion in model.fusions.values():
        fusion.gamma.data.fill_(0.1)
    sum(v.square().mean() for v in model(x).values()).backward()
    for module in [model.encoder, model.queries, model.decoders, model.heads, model.fusions]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
    embedding = model.encoder.backbone.model.patch_embeddings
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in embedding.parameters())
    assert all(f.attention.value.weight.grad.abs().sum() > 0 for f in model.fusions.values())
    model.eval()
    assert not model.fusions.training and not model.queries.training


def test_fresh_fusion_config_does_not_load_task_checkpoint():
    cfg = Config.fromfile('configs/hydrostate_joint_adapterfusion_scratch.py')
    assert cfg.model.baseline_checkpoint is None and not cfg.model.freeze_baseline
    assert cfg.model.encoder.private_rank == 0
    assert cfg.model.encoder.gradient_checkpointing is False
    assert cfg.train_cfg.max_epochs == 25


def test_masked_experts_and_empty_support():
    fusion = ResidualAdapterFusion(4)
    fusion.gamma.data.fill_(1)
    target = torch.randn(1, 4, 2, 2)
    sources = torch.randn(1, 3, 4, 2, 2)
    valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    valid[:, 1] = False
    valid[:, :, 0, 0] = False
    sources[:, 1] = float('nan')
    result = fusion(target, sources, valid, torch.ones(1, 2, 2, dtype=torch.bool))
    torch.testing.assert_close(result[:, :, 0, 0], target[:, :, 0, 0])
    assert torch.isfinite(result).all()
    # The missing expert must not dilute the softmax denominator.
    expected = fusion(target, sources[:, [0, 2]], valid[:, [0, 2]],
                      torch.ones(1, 2, 2, dtype=torch.bool))
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert fusion.attention.value.weight.grad.abs().sum() > 0


def test_frozen_baseline_roundtrip_gradients_and_intervals(tmp_path):
    cfg = config(tmp_path).model
    cfg.encoder.private_rank = 0
    baseline = MODELS.build(cfg).eval()
    path = tmp_path / 'baseline.pth'
    torch.save(dict(state_dict=baseline.state_dict()), path)
    cfg = copy.deepcopy(cfg)
    cfg.update(type='AdapterFusionHydroStateModel', baseline_checkpoint=str(path))
    model = MODELS.build(cfg).train()
    assert not model.encoder.training and not model.decoders.training
    assert model.fusions.training
    x = raw_inputs(model.tasks)
    x['target_time']['soil_moisture'] += 86400
    with torch.no_grad():
        expected = baseline(x)
    actual = model(x)
    for task in actual:
        torch.testing.assert_close(actual[task], expected[task], rtol=0, atol=0)
    sum(v.square().mean() for v in actual.values()).backward()
    assert any(f.gamma.grad.abs() > 0 for f in model.fusions.values())
    assert all(p.grad is None and not p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('fusions.'))
    model.zero_grad(set_to_none=True)
    for f in model.fusions.values():
        f.gamma.data.fill_(0.1)
    sum(v.square().mean() for v in model(x).values()).backward()
    assert all(f.attention.value.weight.grad.abs().sum() > 0 for f in model.fusions.values())
    model.fusion_enabled = False
    for task, value in model(x).items():
        torch.testing.assert_close(value, expected[task], rtol=0, atol=0)
    model.fusion_enabled = True
    restored = MODELS.build(cfg).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    for task, value in model(x).items():
        torch.testing.assert_close(value, restored(x)[task], rtol=0, atol=0)
