"""Task-conditioned low-rank updates with checkpoint-safe task dispatch.

Only the opt-in shared/private encoder installs these layers. The context is
captured at each checkpoint's forward, so recomputation cannot accidentally
use the most recently dispatched task.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TaskContext:
    task: str
    gate_logits: torch.Tensor | None


_CONTEXT = ContextVar("hydrostate_lora_task", default=None)


@contextmanager
def task_context(context):
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class LowRankUpdate(nn.Module):
    def __init__(self, in_features, out_features, rank, scale=1.0):
        super().__init__()
        if type(rank) is not int or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("LoRA scale must be finite and positive")
        self.rank, self.scale = rank, float(scale)
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs):
        return self.b(self.a(inputs)) * self.scale


class SharedPrivateLinear(nn.Module):
    """One frozen linear map, one shared update, and named private updates."""

    def __init__(self, base, block_index, shared_rank, private_ranks, scale=1.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("SharedPrivateLinear requires an nn.Linear")
        if type(shared_rank) is not int or shared_rank < 0:
            raise ValueError("shared_rank must be a nonnegative integer")
        self.base = base.requires_grad_(False)
        self.block_index = block_index
        self.shared = (LowRankUpdate(base.in_features, base.out_features, shared_rank, scale)
                       if shared_rank else None)
        self.private = nn.ModuleDict({
            task: LowRankUpdate(base.in_features, base.out_features, rank, scale)
            for task, rank in private_ranks.items() if rank
        })
        self.tasks = tuple(private_ranks)

    def forward(self, inputs):
        context = _CONTEXT.get()
        if context is None or context.task not in self.tasks:
            raise RuntimeError("LoRA forward requires an explicit configured task")
        output = self.base(inputs)
        if self.shared is not None:
            gate = (context.gate_logits[self.block_index].sigmoid()
                    if context.gate_logits is not None else 1.0)
            output = output + gate * self.shared(inputs)
        if context.task in self.private:
            output = output + self.private[context.task](inputs)
        return output


def checkpointed_task_forward(block):
    """Bind the task at forward time, including when backward is deferred."""
    forward = type(block).forward

    def wrapped(self, *args, **kwargs):
        context = _CONTEXT.get()

        def run(*inner_args, **inner_kwargs):
            with task_context(context):
                return forward(self, *inner_args, **inner_kwargs)

        if torch.is_grad_enabled():
            return checkpoint(run, *args, use_reentrant=False, **kwargs)
        return run(*args, **kwargs)

    # Bind to the module, not a captured original instance: deepcopy must also
    # rebind checkpoint recomputation to the copied module's parameters.
    return MethodType(wrapped, block)


class TaskInputAdapter(nn.Module):
    """Dispatch a task's small input projection without changing legacy adapters."""
    def __init__(self, adapters):
        super().__init__()
        self.adapters = nn.ModuleDict(adapters)

    def forward(self, *args, **kwargs):
        context = _CONTEXT.get()
        if context is None:
            raise RuntimeError("Input adaptation requires an explicit task")
        return self.adapters[context.task](*args, **kwargs)
