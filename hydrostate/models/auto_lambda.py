"""CGC with station-directed one-step learning of product loss coefficients."""
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.site_dataset import SiteDataset
from hydrostate.experiments.auto_lambda import virtual_sgd_step
from hydrostate.registry import MODELS
from .cosa.model import CoSAHydroStateModel


@MODELS.register_module()
class AutoLambdaCGC(CoSAHydroStateModel):
    def __init__(self, meta_plan, meta_interval=10, meta_lr=.001, **kwargs):
        super().__init__(**kwargs)
        self.meta_plan = meta_plan
        self.meta_interval = meta_interval
        self.meta_lr = meta_lr
        self.register_buffer('product_weights', torch.ones(len(self.expert_groups)))
        self.register_buffer('meta_iteration', torch.tensor(0))
        self.register_buffer('meta_m', torch.zeros_like(self.product_weights))
        self.register_buffer('meta_v', torch.zeros_like(self.product_weights))
        self.register_buffer('meta_steps', torch.zeros_like(self.product_weights))
        self.meta_loaders = {}
        self.meta_streams = {}

    def meta_batch(self):
        if not self.meta_loaders:
            cohorts = json.loads(Path(self.meta_plan).read_text())['meta']
            for i, name in enumerate(('sm', 'et')):
                sites = [SiteDataset(x['manifest'], 'train', data_root=x['data_root'])
                         for x in cohorts if x['task'] == name]
                self.meta_loaders[name] = DataLoader(ConcatDataset(sites), shuffle=True,
                    generator=torch.Generator().manual_seed(1324+i), batch_size=1,
                    collate_fn=list, num_workers=0)
        rows = []
        for task, loader in self.meta_loaders.items():
            if task not in self.meta_streams:
                self.meta_streams[task] = iter(loader)
            try:
                rows.extend(next(self.meta_streams[task]))
            except StopIteration:
                self.meta_streams[task] = iter(loader)
                rows.extend(next(self.meta_streams[task]))
        return self.data_preprocessor(hydro_collate(rows), training=False)

    def update_coefficients(self, losses, lr):
        products = []
        for tasks in self.expert_groups.values():
            terms = [losses['loss_'+t] for t in tasks if losses['valid_count_'+t].item() > 0]
            products.append(sum(terms) if terms else None)
        if all(x is None for x in products):
            return
        weights = self.product_weights.detach().clone().requires_grad_(True)
        parameters = {n:p for n,p in self.named_parameters() if p.requires_grad}
        virtual = virtual_sgd_step(parameters, losses['loss_site_sm']+losses['loss_site_et'],
                                   products, weights, lr)
        # A virtual pass must not update running buffers or reuse cached AMP casts.
        self.eval()
        try:
            batch = self.meta_batch()
            with torch.autocast(device_type=weights.device.type, enabled=False):
                meta = torch.func.functional_call(self, virtual, (),
                                                   dict(**batch, mode='loss'))
                objective = meta['loss_site_sm'] + meta['loss_site_et']
                gradient, = torch.autograd.grad(objective, weights)
        finally:
            self.train()
        with torch.no_grad():
            active = weights.new_tensor([x is not None for x in products]).bool()
            self.meta_steps[active] += 1
            self.meta_m[active] = self.meta_m[active]*.9 + gradient[active]*.1
            self.meta_v[active] = self.meta_v[active]*.999 + gradient[active].square()*.001
            first = self.meta_m[active] / (1-.9**self.meta_steps[active])
            second = self.meta_v[active] / (1-.999**self.meta_steps[active])
            self.product_weights[active] = (self.product_weights[active] -
                self.meta_lr * first / (second.sqrt()+1e-8)).clamp(0,2)

    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            batch = self.data_preprocessor(data, training=True)
            losses = self(**batch, mode='loss')
        self.meta_iteration.add_(1)
        if self.meta_iteration.item() % self.meta_interval == 0:
            self.update_coefficients(losses, optim_wrapper.optimizer.param_groups[0]['lr'])
        for i,(group,tasks) in enumerate(self.expert_groups.items()):
            for task in tasks:
                losses['loss_'+task] = self.product_weights[i] * losses['loss_'+task]
            losses['weight_'+group] = self.product_weights[i].detach().clone()
        loss, logs = self.parse_losses(losses)
        optim_wrapper.update_params(loss)
        return logs
