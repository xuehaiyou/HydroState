"""Config-selected task adaptation with independent temporal prediction branches."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS
from hydrostate.tasks import LEGACY_TASKS, task_config

from .hydrostate_model import HydroStateModel
from .multilevel_model import MultiLevelFusionDecoder
from .shared_private_encoder import SharedPrivateOlmoEarthEncoder


@MODELS.register_module()
class SharedPrivateHydroStateModel(HydroStateModel):
    """Shared backbone updates, private input/query/decoder/head parameters.

    ``train_shared=False`` fixes the knowledge source. ``train_tasks`` then
    selects which private branches may change, including a newly added task.
    Full MMEngine checkpoints and smaller version-checked adapter bundles serve
    different purposes: exact resume versus explicit task transfer/composition.
    """

    requires_target_time = True

    def __init__(self, encoder, decoder, heads, query_time=None, train_shared=True,
                 train_tasks=None, adapter_init=None, active_tasks_only=False,
                 branch_init_seed=None, **kwargs):
        if type(active_tasks_only) is not bool:
            raise ValueError("active_tasks_only must be boolean")
        if branch_init_seed is not None and type(branch_init_seed) is not int:
            raise ValueError("branch_init_seed must be an integer")
        if active_tasks_only and kwargs.get("water_soil_coupling"):
            raise ValueError("Task-only execution cannot use cross-task support coupling")
        self.active_tasks_only = active_tasks_only
        if kwargs.get("fraction_groups"):
            raise ValueError("Shared/private v1 requires independent variable heads")
        if kwargs.get("input_fusion"):
            raise ValueError("Shared/private v1 has no shared auxiliary-input fusion")
        # MMEngine's expanded Python config sorts mapping keys. Keep loss buffers
        # and checkpoint task order stable across source and expanded configs.
        specs = task_config(kwargs.get("tasks", LEGACY_TASKS))
        order = [task for task in LEGACY_TASKS if task in specs]
        order += sorted(set(specs) - set(LEGACY_TASKS))
        tasks = kwargs["tasks"] = {task: specs[task] for task in order}
        encoder = copy.deepcopy(encoder)
        if "tasks" in encoder and encoder["tasks"] != tasks:
            raise ValueError("Encoder tasks must come from the model task contract")
        encoder["tasks"] = tasks
        self.branch_configs = dict(decoder=copy.deepcopy(decoder), heads=copy.deepcopy(heads),
                                   query_time=copy.deepcopy(query_time or {}))
        super().__init__(encoder=encoder, decoder=decoder, heads=heads, **kwargs)
        if not isinstance(self.encoder, SharedPrivateOlmoEarthEncoder):
            raise TypeError("SharedPrivateHydroStateModel requires SharedPrivateOlmoEarthEncoder")
        template = self._modules.pop("decoder")
        if not isinstance(template, MultiLevelFusionDecoder):
            raise TypeError("Shared/private v1 requires MultiLevelFusionDecoder")
        self.decoders = nn.ModuleDict({task: copy.deepcopy(template) for task in self.tasks})
        config = dict(query_time or {})
        config.setdefault("type", "TargetTimeQuery")
        self.queries = nn.ModuleDict({
            task: MODELS.build(dict(config, channels=self.encoder.out_channels, tasks=[task]))
            for task in self.tasks
        })
        if branch_init_seed is not None:
            # Identical readout initialization across ranks and sharing modes.
            # Keep the legacy construction unchanged unless explicitly requested.
            for index, task in enumerate(self.tasks):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(branch_init_seed + 1009 * index)
                    self.decoders[task] = MODELS.build(copy.deepcopy(decoder))
                    self.queries[task] = MODELS.build(
                        dict(config, channels=self.encoder.out_channels, tasks=[task]))
                    self.heads[task] = MODELS.build(copy.deepcopy(heads[task]))
        self.set_training_tasks(train_tasks, train_shared=train_shared)
        if adapter_init:
            from .shared_private_transfer import load_adapter_bundle

            sources = adapter_init if isinstance(adapter_init, list) else [adapter_init]
            for source in sources:
                load_adapter_bundle(self, **dict(source))

    def parameter_owner(self, name):
        """Return __shared__ / task name / None (original fixed parameter)."""
        if name.startswith("encoder.backbone.model.") and any(
            f".attn.{target}.shared." in name for target in self.encoder.target_modules
        ):
            return "__shared__"
        for task in self.tasks:
            if (f".private.{task}." in name
                    or name == f"encoder.gates.{task}"
                    or name.startswith(f"encoder.landsat_l2_adapter.adapters.{task}.")
                    or any(name.startswith(f"{part}.{task}.")
                           for part in ("queries", "decoders", "heads"))
                    or name == f"loss_module.log_variances.{task}"):
                return task
        return None

    def set_training_tasks(self, tasks=None, *, train_shared=False):
        selected = tuple(self.tasks if tasks is None else tasks)
        if (not selected or len(set(selected)) != len(selected)
                or not set(selected) <= set(self.tasks)):
            raise ValueError("train_tasks must be nonempty unique configured task names")
        if type(train_shared) is not bool:
            raise ValueError("train_shared must be boolean")
        self.train_tasks, self.train_shared = selected, train_shared
        for name, parameter in self.named_parameters():
            owner = self.parameter_owner(name)
            parameter.requires_grad_(
                train_shared if owner == "__shared__" else owner in selected)
            # A phase transition must not leave an old gradient for AdamW to apply.
            if not parameter.requires_grad:
                parameter.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "train_tasks"):
            for task in self.tasks:
                if task not in self.train_tasks:
                    for branches in (self.queries, self.decoders, self.heads):
                        branches[task].eval()
        return self

    def _validate_queries(self, inputs):
        if "target_time" not in inputs or set(inputs["target_time"]) != set(self.tasks):
            raise ValueError("Shared/private retrieval requires target_time for every task")
        coupling = self.water_soil_coupling
        if coupling is not None and coupling.enabled:
            # This opt-in rule is valid only for an explicitly aligned request.
            # Never silently use another observation date as an SM land mask.
            torch._assert_async(
                (inputs["target_time"]["water"] == inputs["target_time"]["soil_moisture"]).all(),
                "WF-SM coupling requires identical WF and SM query times")

    def _forward_task(self, inputs, task):
        levels = self.encoder(inputs, task=task)
        query = self.queries[task]
        features = tuple(query(level, inputs["target_time"][task], task) for level in levels)
        decoded = self.decoders[task](features)
        if any(size % self.output_scale for size in decoded.shape[-2:]):
            raise ValueError("Decoded grid must be divisible by output_scale")
        return self.heads[task](F.avg_pool2d(decoded, self.output_scale))

    def _forward(self, inputs):
        self._validate_queries(inputs)
        tasks = self.train_tasks if self.active_tasks_only else self.tasks
        return {task: self._forward_task(inputs, task) for task in tasks}

    def loss(self, inputs, data_samples):
        losses = self._loss_from_outputs(self._forward(inputs), data_samples)
        active = False
        for task in (self.train_tasks if self.active_tasks_only else self.tasks):
            # Counts have already been globally reduced, so every DDP rank makes
            # the same decision. Missing tasks get no gradient (not a zero tensor)
            # and therefore no AdamW decay/momentum update.
            if task not in self.train_tasks or losses[f"valid_count_{task}"].item() == 0:
                losses.pop(f"loss_{task}")
            else:
                active = True
        if not active:
            reference = next(self.parameters())
            losses["loss_empty"] = torch.zeros((), device=reference.device, requires_grad=True)
        return losses

    def predict(self, inputs, data_samples=None):
        predictions = super().predict(inputs, data_samples)
        for index, prediction in enumerate(predictions):
            prediction["target_time"] = {
                task: inputs["target_time"][task][index]
                for task in (self.train_tasks if self.active_tasks_only else self.tasks)}
        return predictions
