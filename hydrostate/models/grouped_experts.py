"""Shared retrieval interface for complete grouped expert decoders."""

import torch
from torch import nn
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .hydrostate_model import HydroStateModel
from .mlore.vendor import ConvHead
from .retrieval_model import RetrievalModelMixin


@MODELS.register_module()
class GroupedExpertHydroStateModel(RetrievalModelMixin, HydroStateModel):
    def __init__(self, expert_groups, official_heads=False, branches=None, router=None, **kwargs):
        if branches is not None or router is not None:
            raise ValueError('Complete expert decoders replace CoSA branches and router')
        activations = {task: cfg.get('activation', 'identity')
                       for task, cfg in kwargs['heads'].items()}
        super().__init__(**kwargs)
        self.expert_groups = {name: tuple(tasks) for name, tasks in expert_groups.items()}
        assigned = [task for tasks in self.expert_groups.values() for task in tasks]
        if (not self.expert_groups or any(not tasks for tasks in self.expert_groups.values())
                or len(assigned) != len(set(assigned)) or set(assigned) != set(self.tasks)
                or tuple(self.expert_groups) != self.decoder.groups):
            raise ValueError('Expert groups must partition outputs and match the decoder')
        if self.input_fusion is not None:
            raise ValueError('Expert decoders own observation-quality fusion')
        self.official_heads = official_heads
        self.activations = activations
        if official_heads:
            # One official ConvHead per TASK (Coverage/SM/ET), with 3/1/3 outputs.
            self.heads = nn.ModuleDict({name: ConvHead(self.decoder.out_channels, len(tasks))
                                       for name, tasks in self.expert_groups.items()})

    def _run(self, inputs):
        if set(inputs.get('target_time', {})) != set(self.tasks):
            raise ValueError('Explicit target intervals are required for every task')
        bounds = inputs['target_time'][self.tasks[0]]
        for task in self.tasks[1:]:
            torch._assert_async((inputs['target_time'][task] == bounds).all(),
                                'Grouped expert experiment requires aligned target intervals')
        size = inputs[self.encoder.input_keys[0]].shape[-2:]
        if any(length % self.output_scale for length in size):
            raise ValueError('Input grid must divide by output_scale')
        output_size = tuple(length // self.output_scale for length in size)
        prepared = self.decoder.prepare(self.encoder(inputs), inputs)
        features, valid, auxiliary = self.decoder(prepared, bounds)
        outputs = {}
        for name, tasks in self.expert_groups.items():
            if self.official_heads:
                value = self.heads[name](features[name].masked_fill(~valid, 0))
                value = F.interpolate(value, size=output_size, mode='bilinear',
                                      align_corners=False)
                for task, result in zip(tasks, value.split(1, dim=1)):
                    activation = self.activations[task]
                    if activation == 'sigmoid':
                        result = result.sigmoid()
                    elif activation == 'softplus':
                        result = F.softplus(result)
                    elif activation != 'identity':
                        raise ValueError(f'Unsupported activation {activation}')
                    outputs[task] = result
            else:
                value = F.interpolate(features[name].masked_fill(~valid, 0), size=output_size,
                                      mode='bilinear', align_corners=False)
                outputs.update({task: self.heads[task](value) for task in tasks})
        return self._constrain_outputs(outputs), auxiliary

    def _forward(self, inputs):
        return self._run(inputs)[0]

    def loss(self, inputs, data_samples):
        outputs, auxiliary = self._run(inputs)
        losses = self._loss_from_outputs(outputs, self._land_samples(inputs, data_samples))
        if auxiliary is not None:
            losses['loss_expert_balance'] = auxiliary
        return losses
