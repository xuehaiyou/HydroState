import torch

from hydrostate.models import RegressionHead, SharedUNetDecoder


def test_decoder_and_head_shapes():
    decoder = SharedUNetDecoder(in_channels=16, channels=[16, 8, 4], scale_factor=4)
    head = RegressionHead(in_channels=4, activation="sigmoid")
    output = head(decoder(torch.randn(2, 16, 32, 32)))
    assert output.shape == (2, 1, 128, 128)
    assert torch.all((output >= 0) & (output <= 1))
