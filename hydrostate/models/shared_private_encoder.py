"""Opt-in OlmoEarth parameter adaptation; legacy encoder behavior is untouched."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, resolve_tasks

from .multilevel_temporal_model import MultiLevelTemporalOlmoEarthEncoder
from .shared_private_lora import (
    SharedPrivateLinear,
    TaskContext,
    TaskInputAdapter,
    checkpointed_task_forward,
    task_context,
)


@MODELS.register_module()
class SharedPrivateOlmoEarthEncoder(MultiLevelTemporalOlmoEarthEncoder):
    def __init__(self, tasks=LEGACY_TASKS, shared_rank=8, private_rank=8,
                 target_modules=("q", "v"), gate_init=0.5, lora_scale=1.0,
                 gradient_checkpointing=False, shared_gate_mode="learned", **kwargs):
        # Install checkpoints after adapters, with the task captured explicitly.
        super().__init__(gradient_checkpointing=False, **kwargs)
        self.tasks = tuple(resolve_tasks(tasks))
        if type(shared_rank) is not int or shared_rank < 0:
            raise ValueError("shared_rank must be a nonnegative integer")
        ranks = (dict(private_rank) if isinstance(private_rank, dict)
                 else dict.fromkeys(self.tasks, private_rank))
        if (set(ranks) != set(self.tasks)
                or any(type(r) is not int or r < 0 for r in ranks.values())):
            raise ValueError("private_rank must give a nonnegative integer for every task")
        if not shared_rank and any(not rank for rank in ranks.values()):
            raise ValueError("Each task requires a shared or private adaptation path")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("gate_init must be strictly between zero and one")
        if shared_gate_mode not in ("learned", "fixed_one"):
            raise ValueError("shared_gate_mode must be learned or fixed_one")
        self.shared_gate_mode = shared_gate_mode
        if not target_modules or len(set(target_modules)) != len(target_modules):
            raise ValueError("target_modules must be nonempty and unique")
        if not set(target_modules) <= {"q", "k", "v", "proj"}:
            raise ValueError("Supported attention targets are q, k, v and proj")
        self.shared_rank, self.private_ranks = shared_rank, ranks
        self.target_modules = tuple(target_modules)
        self.lora_scale = float(lora_scale)
        core = self.backbone.model
        core.requires_grad_(False)
        self.gates = nn.ParameterDict({
            task: nn.Parameter(torch.full((len(core.blocks),), math.log(gate_init / (1-gate_init))))
            for task in self.tasks
        } if shared_rank and shared_gate_mode == "learned" else {})
        for index, block in enumerate(core.blocks):
            for name in self.target_modules:
                original = getattr(block.attn, name)
                setattr(block.attn, name, SharedPrivateLinear(
                    original, index, shared_rank, ranks, lora_scale))
            if gradient_checkpointing:
                block.forward = checkpointed_task_forward(block)
        if self.landsat_l2_adapter is not None:
            template = self.landsat_l2_adapter
            self.landsat_l2_adapter = TaskInputAdapter({
                task: copy.deepcopy(template) for task in self.tasks})

    def forward(self, inputs, task=None):
        if task not in self.tasks:
            raise ValueError("Shared/private encoder requires a configured task name")
        context = TaskContext(task, self.gates[task] if task in self.gates else None)
        with task_context(context):
            return super().forward(inputs)

    def train(self, mode=True):
        super().train(mode)
        # Frozen stochastic backbone layers must not drift between tasks/stages.
        # LoRA modules contain only linear layers, so eval does not disable gradients.
        self.backbone.eval()
        return self

    def adaptation_settings(self):
        return dict(shared_rank=self.shared_rank, private_rank=self.private_ranks,
                    shared_gate_mode=self.shared_gate_mode,
                    target_modules=list(self.target_modules), lora_scale=self.lora_scale,
                    out_indices=list(self.out_indices), input_profile=self.input_profile,
                    patch_size=self.patch_size, embedding_channels=self.out_channels)
