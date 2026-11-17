"""Run the authors' MLoRE block on interval-aligned private spatial features."""

from types import SimpleNamespace

from torch import nn

from .vendor import MOEBlock


class MLoRERouter(nn.Module):
    def __init__(self, experts, state_channels=256, channels=64, grid_size=30,
                 ranks=(4, 8, 12, 16), private_rank=8, topk=2, pre_softmax=True):
        super().__init__()
        self.experts = tuple(experts)
        if (not self.experts or channels % 4 or channels < 4
                or not ranks or any(type(rank) is not int or rank < 1 for rank in ranks)
                or not 1 <= topk <= len(ranks) or grid_size < 1 or private_rank < 1):
            raise ValueError("Invalid MLoRE block dimensions or expert selection")
        self.grid_size = grid_size
        options = SimpleNamespace(
            TASKS=SimpleNamespace(NAMES=self.experts), rank_list=list(ranks),
            spe_rank=private_rank, topk=topk, pre_softmax=pre_softmax)
        self.block = MOEBlock(options, channels, grid_size**2, kernel_size=3)

    def forward(self, state, features, valid):
        if set(features) != set(self.experts):
            raise ValueError("MLoRE features disagree with task experts")
        if state.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError("MLoRE spatial gate requires its configured feature grid")
        return {task: self.block(features[task].masked_fill(~valid, 0), task)[0].masked_fill(
            ~valid, 0) for task in self.experts}
