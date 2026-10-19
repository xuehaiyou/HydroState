"""Trainable SR projection into the pretrained Landsat input embedding."""

import torch
from torch import nn


class LandsatL2Adapter(nn.Module):
    """Seven SR bands -> eleven learned features; these are not synthetic L1 bands.

    Fixed physical scaling conditions optimization, without estimating statistics
    from held-out imagery. Negative water reflectance is retained. The initial
    seven diagonal weights preserve spectral ordering; remaining features start
    at zero and are learned during fine-tuning, with no thermal input or label.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.tensor(0.2))
        self.register_buffer("scale", torch.tensor(0.2))
        self.projection = nn.Conv2d(7, 11, 1)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            for band in range(7):
                self.projection.weight[band, band, 0, 0] = 1

    def forward(self, image, valid):
        pixels = valid.bool().all(dim=1, keepdim=True)
        scaled = ((image.float() - self.center) / self.scale).masked_fill(~pixels, 0)
        return self.projection(scaled).masked_fill(~pixels, 0)
