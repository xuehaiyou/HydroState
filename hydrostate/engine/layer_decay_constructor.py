"""MMEngine implementation of the OlmoEarth layer-wise LR decay recipe."""

from __future__ import annotations

import re
from collections import defaultdict

import torch
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from hydrostate.registry import OPTIM_WRAPPER_CONSTRUCTORS

_BLOCK_PATTERN = re.compile(r"(?:blocks|layers)\.(\d+)(?:\.|$)")


def olmoearth_layer_id(name: str, num_layers: int) -> int:
    """Map a HydroState parameter name to an LLRD depth.

    Patch/input projections use depth 0, transformer block ``i`` uses ``i+1``,
    and final encoder norms use ``num_layers``. The shared decoder and heads
    use depth ``num_layers + 1`` and therefore the unscaled base LR.
    """
    if name.startswith(("decoder.", "heads.", "loss_module.")):
        return num_layers + 1
    match = _BLOCK_PATTERN.search(name)
    if match:
        return min(int(match.group(1)) + 1, num_layers)
    if name.startswith("encoder.") and any(
        token in name for token in ("final_norm", "post_norm", ".norm.", ".norm.weight")
    ):
        return num_layers
    if name.startswith("encoder."):
        return 0
    return num_layers + 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class OlmoEarthLayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Build AdamW groups using OlmoEarth's recommended LLRD settings."""

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None) -> None:
        if paramwise_cfg is None:
            raise ValueError("paramwise_cfg with num_layers and layer_decay_rate is required")
        self.num_layers = int(paramwise_cfg.get("num_layers", 12))
        self.layer_decay_rate = float(paramwise_cfg.get("layer_decay_rate", 0.65))
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 < self.layer_decay_rate <= 1:
            raise ValueError("layer_decay_rate must be in (0, 1]")
        # DefaultOptimWrapperConstructor only calls add_params for a truthy
        # paramwise_cfg. The custom options are consumed above; this private
        # sentinel deliberately enables our add_params implementation.
        super().__init__(optim_wrapper_cfg, paramwise_cfg={"_hydrostate_llrd": True})

    @staticmethod
    def _no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
        lowered = name.lower()
        return parameter.ndim == 1 or name.endswith(".bias") or any(
            key in lowered for key in ("norm", "pos_embed", "position_embedding")
        )

    def add_params(self, params: list[dict], module: torch.nn.Module, prefix: str = "") -> None:
        grouped: dict[tuple[int, bool], list[torch.nn.Parameter]] = defaultdict(list)
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            layer_id = olmoearth_layer_id(name, self.num_layers)
            no_decay = self._no_decay(name, parameter)
            grouped[(layer_id, no_decay)].append(parameter)

        max_layer_id = self.num_layers + 1
        for (layer_id, no_decay), parameters in sorted(grouped.items()):
            lr_scale = self.layer_decay_rate ** (max_layer_id - layer_id)
            group = {
                "params": parameters,
                "lr": self.base_lr * lr_scale,
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else self.base_wd,
                "group_name": f"layer_{layer_id}_{'no_decay' if no_decay else 'decay'}",
            }
            params.append(group)
            print_log(
                f"{group['group_name']}: lr={group['lr']:.3e}, "
                f"weight_decay={group['weight_decay']}, tensors={len(parameters)}",
                logger="current",
            )
