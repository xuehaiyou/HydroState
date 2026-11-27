"""Total effective product supervision for the trusted-gradient comparison."""

from pathlib import Path

import torch
from mmengine.hooks import Hook

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.registry import HOOKS


@HOOKS.register_module()
class ProductWeightCountsHook(Hook):
    """Accumulate actual valid and changed product observations over the whole epoch."""

    def before_train_epoch(self, runner):
        self.totals = {}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in outputs.items():
            if name.startswith(
                (
                    "valid_count_",
                    "guided_count_",
                    "trusted_count_",
                    "eligible_batches_",
                    "paired_batches_",
                )
            ):
                count = value.detach().to(torch.int64)
                if name not in self.totals:
                    self.totals[name] = count.clone()
                else:
                    self.totals[name] += count

    def after_train_epoch(self, runner):
        values = torch.stack(list(self.totals.values())).tolist()
        atomic_json(
            Path(runner.work_dir) / f"weight_counts_epoch_{runner.epoch + 1}.json",
            dict(zip(self.totals, values)),
        )
