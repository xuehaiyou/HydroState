"""Same-variable trusted gradients guide the relative weights of product observations."""

import torch

from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.models.manual_cover_supervision import COVER, manual_cover_loss
from hydrostate.models.site_supervision import station_values
from hydrostate.registry import MODELS

from .foundation_v2 import FoundationStageModelV2


def readout_scores(errors, trusted, parameters):
    """Evaluate J_errors @ grad(trusted) without constructing the observation Jacobian."""
    direction = torch.autograd.grad(trusted, parameters, retain_graph=True)
    probe = torch.zeros_like(errors, requires_grad=True)
    vjp = torch.autograd.grad(
        errors, parameters, grad_outputs=probe, create_graph=True, retain_graph=True
    )
    return torch.autograd.grad(
        vjp, probe, grad_outputs=tuple(t.detach() for t in direction), retain_graph=False
    )[0].detach()


@torch.no_grad()
def relative_weights(scores, quality):
    """Bound relative preference and preserve the total pre-existing quality mass."""
    peak = scores.abs().amax()
    factor = torch.exp(scores / peak.clamp_min(torch.finfo(scores.dtype).tiny))
    proposed = quality * factor
    weights = proposed * (quality.sum() / proposed.sum().clamp_min(torch.finfo(scores.dtype).tiny))
    return torch.where(scores.amax() == scores.amin(), quality, weights)


def standard_errors(prediction, target, valid, std):
    diff = prediction.float().masked_fill(~valid, 0) - target.float().masked_fill(~valid, 0)
    return (diff / std).square()


@MODELS.register_module()
class TrustedGradientBDTA(FoundationStageModelV2):
    def _loss_from_outputs(self, outputs, data_samples, auxiliary=None, auxiliary_weights=None):
        support = self.water_soil_coupling.prepare(outputs, data_samples, supervision=True)
        manual_valid = torch.stack([data_samples[f"manual_{t}_valid"] for t in COVER]).any(0)
        manual_valid = manual_valid & data_samples["prediction_valid"]
        trusted = {
            "coverage": (
                manual_cover_loss(outputs, data_samples, weight=1),
                manual_valid.flatten(1).any(1).sum(),
            )
        }
        for name, (prediction, target, valid) in station_values(outputs, data_samples).items():
            std = self.site_objective.target_std[self.site_objective.names.index(name)]
            count = valid.sum()
            trusted[name.removeprefix("site_")] = (
                standard_errors(prediction, target, valid, std).sum() / count.clamp_min(1),
                count,
            )
        losses = {}
        for group, tasks in self.expert_groups.items():
            trust, anchors = trusted[group]
            losses[f"loss_trusted_{group}"] = (2 / 9) * trust
            losses[f"trusted_count_{group}"] = anchors.detach().float()
            errors, qualities, counts = [], [], []
            for task in tasks:
                pred, target, valid = supervision_values(
                    task, outputs[task], data_samples, support, self.task_specs
                )
                std = self.loss_module.target_std[self.loss_module.tasks.index(task)]
                value = standard_errors(pred, target, valid, std)
                quality = data_samples.get(task + "_weight")
                quality = (
                    valid.float()
                    if quality is None
                    else quality.detach().float().expand_as(valid).masked_fill(~valid, 0)
                )
                errors.append(value.reshape(-1))
                qualities.append(quality.reshape(-1))
                counts.append(valid.sum())
            # One direction per group, comparing only observations of the same output.
            eligible = torch.stack([(q > 0).sum() >= 2 for q in qualities]) & (anchors > 0)
            selected = [i for i, flag in enumerate(eligible.tolist()) if flag]
            new_weights = list(qualities)
            if selected:
                vectors = [errors[i] for i in selected]
                scores = readout_scores(
                    torch.cat(vectors), trust, tuple(self.heads[group].parameters())
                )
                for i, scores_i in zip(selected, scores.split([v.numel() for v in vectors])):
                    q = qualities[i]
                    valid = q > 0
                    w = q.clone()
                    w[valid] = relative_weights(scores_i[valid], q[valid])
                    new_weights[i] = w
            means = []
            for j, (task, err, count, q, w) in enumerate(
                zip(tasks, errors, counts, qualities, new_weights)
            ):
                means.append((err * w).sum() / count.clamp_min(1))
                losses[f"valid_count_{task}"] = count.float()
                losses[f"paired_batches_{task}"] = ((count > 0) & (anchors > 0)).float()
                losses[f"eligible_batches_{task}"] = eligible[j].float()
                losses[f"guided_count_{task}"] = (
                    (~torch.isclose(w, q, rtol=1e-5, atol=1e-8)).sum().float()
                )
            present = (torch.stack(counts) > 0).sum()
            losses[f"loss_product_{group}"] = torch.stack(means).sum() / (9 * present.clamp_min(1))
        return losses
