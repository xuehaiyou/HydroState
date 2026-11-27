"""Quality-masked squared errors in training-set standard deviation units."""

import torch
from torch import nn

from hydrostate.normalization import load_target_statistics
from hydrostate.registry import MODELS
from hydrostate.tasks import resolve_tasks


@MODELS.register_module()
class FixedStandardizedMSE(nn.Module):
    def __init__(self, target_statistics, tasks):
        super().__init__()
        self.tasks = tuple(resolve_tasks(tasks))
        self.target_statistics = load_target_statistics(target_statistics, tasks)
        scales = [self.target_statistics["tasks"][t]["std"] for t in self.tasks]
        self.register_buffer("target_std", torch.tensor(scales, dtype=torch.float32))
        digest = bytes.fromhex(self.target_statistics["training_fingerprint"])
        self.register_buffer("training_fingerprint", torch.tensor(list(digest), dtype=torch.uint8))

    def local_sum(self, task, prediction, target, valid, weight=None):
        prediction, target, valid = torch.broadcast_tensors(prediction, target, valid.bool())
        delta = prediction.float().masked_fill(~valid, 0) - target.float().masked_fill(~valid, 0)
        errors = (delta / self.target_std[self.tasks.index(task)]).square()
        if weight is not None:
            errors = errors * weight.detach().float().expand_as(errors).masked_fill(~valid, 0)
        return errors.sum()
