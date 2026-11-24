import copy

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.models.cosa.faar import RankedDirection
from hydrostate.models.cosa.nexusflow import AffineCoupling
from hydrostate.models.cosa.rutal import ResidualExpertFFN


def test_rank_direction_matches_dora_and_restores_active_slots():
    torch.manual_seed(15)
    base = nn.Linear(16, 12)
    adapter = RankedDirection(base.weight, rank=8).eval()
    with torch.no_grad():
        adapter.b.normal_()
    x = torch.randn(2, 11, 16)

    def compare():
        ids = adapter.order[: adapter.active]
        direction = base.weight + 4 * adapter.b[:, ids] @ adapter.a[ids]
        dense = F.normalize(direction, dim=1) * adapter.magnitude[:, None]
        torch.testing.assert_close(adapter(base, x), F.linear(x, dense, base.bias))

    compare()
    adapter(base, x).sum().backward()
    adapter.score()
    assert adapter.importance.sum() > 0
    adapter.importance.zero_()
    adapter.importance[6] = 1
    adapter.shrink()
    assert adapter.active == 2 and adapter.order[0] == 6
    compare()
    restored = RankedDirection(base.weight, rank=8).eval()
    restored.load_state_dict(adapter.state_dict())
    assert restored.active == 2
    torch.testing.assert_close(restored(base, x), adapter(base, x))


def test_nexus_coupling_inverse():
    coupling = AffineCoupling(20)
    x = torch.randn(4, 20)
    torch.testing.assert_close(coupling(coupling(x), inverse=True), x)


def test_rutal_upcycling_starts_at_pretrained_function():
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.attention import Mlp

    base = Mlp(32, 64).eval()
    upcycled = ResidualExpertFFN(copy.deepcopy(base)).eval()
    x = torch.randn(2, 11, 32)
    torch.testing.assert_close(base(x), upcycled(x))
