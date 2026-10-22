"""Isolated components for paired SM causal experiments."""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from mmengine.dist import get_dist_info

from hydrostate.legacy.dataset import HydroStateZarrDataset
from hydrostate.losses.uncertainty_nll import UncertaintyWeightedNLL, masked_standardized_mse_sum
from hydrostate.models.hydrostate_model import HydroStateModel
from hydrostate.models.olmoearth_encoder import OlmoEarthEncoder, INPUT_KEYS, MODALITIES
from hydrostate.normalization import LEGACY_TASK_NAMES as TASKS
from hydrostate.registry import DATASETS, DATA_SAMPLERS, MODELS


def temporal_signature(times, valid, tolerance, order):
    """Mirror rslearn's greedy time alignment; key only on token layout, not dates.

    Equal keys mean equal modality presence, aligned temporal masks and tensor
    shapes. Each group therefore follows the same path as singleton inference.
    """
    selected, mapping = [], {}
    for mi in order:
        if not any(valid[t][mi] for t in range(len(valid))):
            continue
        assigned = []
        for ti in range(len(times)):
            if not valid[ti][mi]:
                continue
            current = times[ti][mi]
            best, distance = None, tolerance
            for previous in selected:
                delta = abs(current - previous)
                if previous not in assigned and delta < distance:
                    best, distance = previous, delta
            if best is None:
                if current in selected:
                    raise ValueError('Duplicate modality timestamp')
                selected.append(current)
                best = current
            assigned.append(best)
        mapping[mi] = assigned
    selected.sort()
    return tuple((mi, tuple(selected.index(v) for v in mapping[mi])) for mi in mapping)


@MODELS.register_module()
class CausalOlmoEarthEncoder(OlmoEarthEncoder):
    """Group identical token layouts, avoiding batch-dependent padding/pooling.

    Uses the installed wrapper without changing pretrained weights or site-packages.
    This is an opt-in execution correction, not a claim to have fixed every
    upstream issue. The GPU consistency gate must pass before training.
    """
    def __init__(self, time_condition='none', **kwargs):
        super().__init__(**kwargs)
        if time_condition not in ('none', 'relative', 'zero'):
            raise ValueError(time_condition)
        self.time_condition = time_condition
        if time_condition != 'none':
            # Adding the optional branch must not change decoder/head RNG draws.
            with torch.random.fork_rng(devices=[]):
                self.time_adapter = torch.nn.Sequential(
                    torch.nn.Linear(24, 64), torch.nn.GELU(),
                    torch.nn.Linear(64, self.out_channels))

    def forward(self, inputs):
        from rslearn.models.olmoearth_pretrain.model import MODALITY_NAMES
        order = [MODALITIES.index(m) for m in MODALITY_NAMES if m in MODALITIES]
        times = inputs['observation_time'].detach().cpu().tolist()
        valid = inputs['sensor_valid'].detach().cpu().tolist()
        groups = defaultdict(list)
        for i, (t, v) in enumerate(zip(times, valid)):
            groups[temporal_signature(t, v, self.backbone.timestamp_error_tolerance.total_seconds(), order)].append(i)
        outputs, indices = [], []
        for members in groups.values():
            # Prepared index offsets refer to the full batch. Rebuild metadata
            # for each group from its CPU controls instead of slicing offsets.
            selected = {k: v[members] for k, v in inputs.items()
                        if not k.startswith('_temporal_')}
            outputs.append(super().forward(selected))
            indices.extend(members)
        features = torch.cat(outputs)[torch.argsort(torch.tensor(indices, device=outputs[0].device))]
        if self.time_condition != 'none':
            available = inputs['sensor_valid'].to(device=features.device, dtype=torch.float32)
            times_tensor = inputs['observation_time'].to(features.device)
            target_time = inputs['sm_target_time'].to(features.device)
            delta = (times_tensor - target_time[:, None, None]).float() / (86400 * 63)
            delta = delta.clamp(-2, 2) * available
            if self.time_condition == 'zero':
                delta = delta * 0
            condition = torch.cat((delta.flatten(1), available.flatten(1)), dim=1)
            features = features + self.time_adapter(condition)[:, :, None, None]
        return features


@MODELS.register_module()
class CausalNLL(UncertaintyWeightedNLL):
    def __init__(self, objective='nll', **kwargs):
        super().__init__(**kwargs)
        if objective not in ('nll', 'mse'):
            raise ValueError(objective)
        self.objective = objective
        if objective == 'mse':
            for value in self.log_variances.values():
                value.data.fill_(math.log(0.5))
                value.requires_grad_(False)

    def local_sum(self, task, prediction, target, valid):
        if self.objective == 'mse':
            # Preserve the MSE objective in both standalone and combined-head
            # normalization paths (there is no Gaussian constant in this mode).
            return masked_standardized_mse_sum(
                prediction, target, valid, self.target_std[TASKS.index(task)])
        return super().local_sum(task, prediction, target, valid)


@MODELS.register_module()
class CausalHydroStateModel(HydroStateModel):
    def __init__(self, active_tasks=TASKS, freeze_encoder=False, **kwargs):
        super().__init__(**kwargs)
        if not active_tasks or not set(active_tasks) <= set(TASKS):
            raise ValueError('Invalid active tasks')
        self.active_tasks = tuple(active_tasks)
        for task in TASKS:
            if task not in self.active_tasks:
                self.heads[task].requires_grad_(False)
                self.loss_module.log_variances[task].requires_grad_(False)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.freeze_encoder = freeze_encoder

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def loss(self, inputs, data_samples):
        result = super().loss(inputs, data_samples)
        # Inactive heads have no optimizer state or weight decay, and their loss
        # is absent rather than a zero-weight graph attached to shared features.
        for task in TASKS:
            if task not in self.active_tasks:
                del result[f'loss_{task}']
        return result


def filter_temporal_inputs(inputs, target_day, variant):
    """Use day-resolution SM target and fixed common support for C comparisons."""
    inputs['sm_target_time'] = torch.tensor(target_day, dtype=torch.int64)
    if variant == 'original':
        return inputs, None
    if variant not in ('history', 'recent'):
        raise ValueError(variant)
    # SM product timestamp is a day marker: allow that UTC day, not later days.
    allowed = inputs['sensor_valid'].bool() & (inputs['observation_time'] < target_day + 86400)
    recent = torch.zeros_like(allowed)
    for mi in range(allowed.shape[1]):
        choices = torch.where(allowed[:, mi])[0]
        if len(choices):
            latest = choices[inputs['observation_time'][choices, mi].argmax()]
            recent[latest, mi] = True
    if not recent.any():
        raise ValueError('No causal input available for SM target day')
    # Shared spatial support is derived from recent observations for ALL C arms.
    support = []
    for mi, key in enumerate(INPUT_KEYS):
        mask = inputs[f'{key}_valid'].bool()
        shape = (len(recent),) + (1,) * (mask.ndim - 1)
        support.append((mask & recent[:, mi].reshape(shape)).any(dim=0))
        kept = recent[:, mi] if variant == 'recent' else allowed[:, mi]
        inputs[f'{key}_valid'] = mask & kept.reshape(shape)
        inputs[key] = inputs[key] * kept.reshape((len(kept),) + (1,) * (inputs[key].ndim - 1))
    inputs['sensor_valid'] = recent if variant == 'recent' else allowed
    return inputs, torch.stack(support).any(dim=0)


@DATASETS.register_module()
class CausalDataset(HydroStateZarrDataset):
    def __init__(self, temporal_variant='original', **kwargs):
        super().__init__(**kwargs)
        self.temporal_variant = temporal_variant

    def __getitem__(self, index):
        # Base augmentation already transformed all spatial arrays consistently.
        item = super().__getitem__(index)
        row = self.table.iloc[index]
        label = json.loads(row.labels).get('sm')
        target = int(label['system:time_start'] // 1000) if label else int(pd.Timestamp(row.target_time, tz='UTC').timestamp())
        item['inputs'], common = filter_temporal_inputs(item['inputs'], target, self.temporal_variant)
        if common is not None:
            item['data_samples']['causal_common_support'] = common
        return item


from hydrostate.models.data_preprocessor import HydroDataPreprocessor


@MODELS.register_module()
class CausalPreprocessor(HydroDataPreprocessor):
    def forward(self, data, training=False):
        result = super().forward(data, training)
        samples = result['data_samples']
        if 'causal_common_support' in samples:
            common = torch.nn.functional.avg_pool2d(samples.pop('causal_common_support').float(), 3).eq(1)
            samples['prediction_valid'] &= common
        return result


@DATA_SAMPLERS.register_module()
class FixedGlobalBatchSampler(Sampler):
    """Saved global batch rows, sharded contiguously across DDP ranks."""
    def __init__(self, dataset, schedule, seed=None, **kwargs):
        self.rank, self.world = get_dist_info()
        self.rows = np.load(schedule, mmap_mode='r')
        if self.rows.ndim != 2 or self.rows.shape[1] % self.world:
            raise ValueError('Global batch is not divisible by world size')
        if self.rows.min() < 0 or self.rows.max() >= len(dataset):
            raise ValueError('Schedule indices outside dataset')
        per_rank = self.rows.shape[1] // self.world
        self.local = self.rows[:, self.rank * per_rank:(self.rank + 1) * per_rank]

    def __iter__(self):
        return iter(self.local.reshape(-1).tolist())

    def __len__(self):
        return self.local.size

    def set_epoch(self, epoch):
        if epoch:
            raise RuntimeError('Fixed schedule exhausted; do not silently replay it')
