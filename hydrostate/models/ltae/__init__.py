"""OlmoEarth exit pooling with U-TAE's L-TAE module (upstream 987874e27)."""
from hydrostate.registry import MODELS

from ..cosa.state import CoSAStateFusion
from .ltae import LTAE2d


class IntervalLTAE(LTAE2d):
    def forward(self, value, valid, times, bounds, quality):
        positions = (times.double() - bounds.double().mean(-1, keepdim=True)) / 86400
        images = value.masked_fill(~valid[..., None], 0).permute(0, 1, 4, 2, 3)
        pooled = super().forward(images, positions.float(), pad_mask=~valid)
        return pooled.masked_fill(~valid.any(1)[:, None], 0)


@MODELS.register_module()
class LTAEStateFusion(CoSAStateFusion):
    def __init__(self, channels=256, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.pooler = IntervalLTAE(in_channels=channels, n_head=16, d_model=channels,
                                  mlp=[channels, channels])
