"""Optimizer groups for the opt-in shared/private model, without legacy LLRD."""

import math
from collections import defaultdict

from mmengine.optim import DefaultOptimWrapperConstructor

from hydrostate.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SharedPrivateOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        options = dict(paramwise_cfg or {})
        allowed = {"shared_lr_mult", "private_lr_mult", "gate_lr_mult"}
        if set(options) - allowed:
            raise ValueError(f"Unknown shared/private optimizer options: {set(options) - allowed}")
        self.multipliers = {key: options.get(key, 1.0) for key in allowed}
        if any(not math.isfinite(v) or v <= 0 for v in self.multipliers.values()):
            raise ValueError("Learning-rate multipliers must be positive and finite")
        super().__init__(optim_wrapper_cfg, paramwise_cfg={"shared_private": True})

    def add_params(self, params, module, prefix=""):
        if not hasattr(module, "parameter_owner"):
            raise TypeError("SharedPrivateOptimWrapperConstructor needs the shared/private model")
        groups = defaultdict(list)
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            owner = module.parameter_owner(name)
            if owner is None:
                raise ValueError(f"Unowned trainable parameter: {name}")
            kind = ("gate" if name.startswith("encoder.gates.") else
                    "shared" if owner == "__shared__" else "private")
            no_decay = parameter.ndim <= 1 or name.startswith("loss_module.")
            groups[(owner, kind, no_decay)].append(parameter)
        for (owner, kind, no_decay), values in sorted(groups.items()):
            params.append(dict(params=values, lr=self.base_lr * self.multipliers[f"{kind}_lr_mult"],
                               weight_decay=0.0 if no_decay else self.base_wd,
                               group_name=f"{owner}_{kind}_{'no_decay' if no_decay else 'decay'}"))
