"""Shared-parameter gradient surgery on active product and station objectives."""
import torch

from hydrostate.registry import MODELS
from .cosa.model import CoSAHydroStateModel


def combination(gram, method):
    n = len(gram)
    eye = torch.eye(n, device=gram.device, dtype=gram.dtype)
    if method == 'pcgrad':
        rows = []
        for i in range(n):
            row = eye[i]
            for j in torch.randperm(n).tolist():
                if j != i:
                    projection = (row @ gram[:, j]) / gram[j, j].clamp_min(1e-30)
                    row = row - projection.clamp_max(0) * eye[j]
            rows.append(row)
        return torch.stack(rows).sum(0)
    values, vectors = torch.linalg.eigh(gram)
    keep = values > values.max() * n * torch.finfo(gram.dtype).eps
    if not keep.any():
        return eye.sum(0)
    basis = vectors[:, keep]
    return basis @ ((values[keep].min() / values[keep]).sqrt() * basis.sum(0))


@MODELS.register_module()
class GradientCGC(CoSAHydroStateModel):
    def __init__(self, gradient_method, **kwargs):
        super().__init__(**kwargs)
        if gradient_method not in ('pcgrad', 'aligned'):
            raise ValueError(gradient_method)
        self.gradient_method = gradient_method

    def loss(self, inputs, data_samples):
        losses = super().loss(inputs, data_samples)
        groups = [*self.expert_groups.values(), ('site_sm',), ('site_et',)]
        objectives = []
        for tasks in groups:
            terms = [losses['loss_' + t] for t in tasks if 'loss_' + t in losses
                     and losses['valid_count_' + t].item() > 0]
            if terms:
                objectives.append(sum(terms))
        if len(objectives) < 2 or not torch.is_grad_enabled():
            return losses
        params = [p for module in (self.encoder, self.decoder, self.router.shared)
                  for p in module.parameters() if p.requires_grad]
        grads = [torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
                 for x in objectives]
        with torch.autocast(device_type=params[0].device.type, enabled=False):
            gram = torch.stack([torch.stack([sum((a.flatten().float() @ b.flatten().float())
                for a, b in zip(left, right) if a is not None and b is not None)
                for right in grads]) for left in grads])
            weights = combination(gram.detach(), self.gradient_method) - 1
            # Zero-valued surrogate changes only the shared gradient. MMEngine
            # applies AMP scaling and accumulation to the complete objective.
            delta = sum(((p - p.detach()) * sum(w * g.detach() for w, g in zip(weights, column)
                        if g is not None)).sum() for p, column in zip(params, zip(*grads))
                        if any(g is not None for g in column))
        losses['loss_shared_gradient_adjustment'] = delta
        return losses
