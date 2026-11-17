import pytest
import torch
from mmengine.config import Config

from hydrostate.models.mlore.routing import MLoRERouter


def test_official_block_sparse_gates_gradient_and_mask():
    model = MLoRERouter(('sm', 'et'), channels=8, grid_size=3,
                        ranks=(2, 4, 6), private_rank=2, topk=2).eval()
    state = torch.randn(2, 16, 3, 3)
    valid = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    valid[0] = False
    features = {task: torch.randn(2, 8, 3, 3, requires_grad=True) for task in ('sm', 'et')}
    value, _, weights = model.block(features['sm'], 'sm')
    assert ((weights > 0).sum(1) == 2).all()
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    outputs = model(state, features, valid)
    assert not outputs['sm'][0].any()
    torch.testing.assert_close(outputs['sm'][1], value[1])
    outputs['sm'].square().sum().backward()
    assert features['sm'].grad.abs().sum() > 0
    assert features['et'].grad is None
    assert model.block.share_conv.weight.grad.abs().sum() > 0
    assert model.block.conv2['sm'].W.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='grid'):
        model(state[..., :2], features, valid)


def test_mlore_only_changes_composition_configuration():
    base = Config.fromfile('configs/hydrostate_joint.py')
    candidate = Config.fromfile('configs/hydrostate_joint_mlore.py')
    candidate.model.router = base.model.router
    candidate.work_dir = base.work_dir
    assert candidate.to_dict() == base.to_dict()
