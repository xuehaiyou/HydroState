"""U-TAE's grouped temporal attention skip decoder with native retrieval inputs."""
import torch
from torch import nn

from hydrostate.data.retrieval.contract import INPUT_GROUPS, PROFILE, QUALITY_CHANNELS, SENSORS
from hydrostate.registry import MODELS
from ..unet import CentralWindowModel, sequence_parts
from .utae import UTAE


@MODELS.register_module()
class UTAEInput(nn.Module):
    def __init__(self, input_profile=PROFILE):
        super().__init__()
        if input_profile != PROFILE:
            raise ValueError('Expected the five-slot retrieval profile')
        self.input_keys = tuple(SENSORS)

    def forward(self, inputs):
        x = torch.cat(sequence_parts(inputs), dim=2)
        bounds = inputs['slot_bounds'].double()
        days = (bounds.mean(-1) - bounds[:, 2:3].mean(-1)) / 86400
        return x, days.float()


@MODELS.register_module()
class UTAEDecoder(UTAE):
    def __init__(self):
        width = sum(SENSORS.values()) + len(INPUT_GROUPS) * (QUALITY_CHANNELS + 1)
        super().__init__(input_dim=width, out_conv=[32, 64])

    def forward(self, packed):
        x, days = packed
        return super().forward(x, batch_positions=days)


MODELS.register_module(module=CentralWindowModel)
