"""Observation-time queries over unpooled OlmoEarth spatial/temporal tokens.

This is a retrospective retrieval/reconstruction model. Its encoder can attend
to all supplied observations; a late query mask would not make it causal.
"""

from datetime import timedelta

import torch
from torch.nn import functional as F

from hydrostate.registry import MODELS

from .hydrostate_model import HydroStateModel
from .multilevel_encoder import multilevel_features
from .olmoearth_encoder import INPUT_KEYS, MODALITIES, OlmoEarthEncoder


@MODELS.register_module()
class TemporalOlmoEarthEncoder(OlmoEarthEncoder):
    """Keep one feature map per real acquisition, without the 15-day merge."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Exact epoch-second observations may share a slot; nearby dates may not.
        self.backbone.timestamp_error_tolerance = timedelta(microseconds=1)
        self.backbone.token_pooling = False
        core = self.backbone.model
        if core.use_flash_attn:
            raise ValueError("Temporal encoder requires masked PyTorch SDPA attention")
        core.project_and_aggregate.requires_grad_(False)

    def forward(self, inputs):
        return self.forward_levels(inputs, (len(self.backbone.model.blocks) - 1,))[0]

    def forward_levels(self, inputs, out_indices):
        """Align observations once and retain their tokens at selected encoder depths."""
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue

        context = self._make_context(inputs)
        if self.backbone.normalize:
            for image in context.inputs:
                self.backbone.normalizer(image, {})
        sample, present, device = self.backbone._prepare_modality_inputs(context)
        present = [name for name in MODALITIES if name in present]
        valid_cpu = inputs["sensor_valid"].detach().cpu().bool().tolist()
        times_cpu = inputs["observation_time"].detach().cpu().long().tolist()
        unions = [sorted({stamp for stamps, flags in zip(times, validity)
                          for stamp, flag in zip(stamps, flags) if flag})
                  for times, validity in zip(times_cpu, valid_cpu)]
        steps = inputs["sensor_valid"].shape[1]
        aligned_steps = sample.timestamps.shape[1]
        gather = {}
        # Propagate per-pixel observation masks into the backbone as well as pooling.
        for modality in present:
            mi = MODALITIES.index(modality)
            key = INPUT_KEYS[mi]
            original_slots, aligned_slots = [], []
            for times, validity, union in zip(times_cpu, valid_cpu, unions):
                source = {times[t][mi]: t for t in range(steps) if validity[t][mi]}
                original_slots.append([union.index(times[t][mi]) if validity[t][mi] else 0
                                       for t in range(steps)])
                aligned_slots.append([source.get(stamp, 0) for stamp in union]
                                     + [0] * (aligned_steps - len(union)))
            gather[modality] = torch.tensor(original_slots, device=device)
            aligned = torch.tensor(aligned_slots, device=device)
            pixel_valid = inputs[key + "_valid"].bool()
            if pixel_valid.ndim == 5:
                pixel_valid = pixel_valid[:, :, 0]
            h, w = pixel_valid.shape[-2:]
            pixels = pixel_valid.gather(1, aligned[:, :, None, None].expand(-1, -1, h, w))
            mask = getattr(sample, modality + "_mask")
            mask = mask.masked_fill(~pixels.permute(0, 2, 3, 1)[..., None], MaskValue.MISSING.value)
            sample = sample._replace(**{modality + "_mask": mask})

        core = self.backbone.model
        levels = multilevel_features(
            core, sample, self.patch_size, out_indices, retain_tokens=True)
        return tuple(self._restore_observations(tokens, masks, present, gather, inputs, device)
                     for tokens, masks in levels)

    @staticmethod
    def _restore_observations(tokens, masks, present, gather, inputs, device):
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue

        features, validity, observations = [], [], []
        for modality in present:
            mi = MODALITIES.index(modality)
            values = tokens[modality]  # B,H,W,T,S,C
            valid = masks[modality + "_mask"] == MaskValue.ONLINE_ENCODER.value
            count = valid.sum(dim=4)
            values = values.masked_fill(~valid[..., None], 0).sum(dim=4)
            values = values / count.clamp(min=1)[..., None]
            values = values.permute(0, 3, 4, 1, 2)  # B,T,C,H,W
            valid = (count > 0).permute(0, 3, 1, 2)
            index = gather[modality]
            _, _, channels, h, w = values.shape
            values = values.gather(1, index[:, :, None, None, None].expand(-1, -1, channels, h, w))
            valid = valid.gather(1, index[:, :, None, None].expand(-1, -1, h, w))
            sensor = inputs["sensor_valid"][:, :, mi].to(device=device, dtype=torch.bool)
            valid = valid & sensor[:, :, None, None]
            features.append(values.masked_fill(~valid[:, :, None], 0))
            validity.append(valid)
            stamp = inputs["observation_time"][:, :, mi].to(device).masked_fill(~sensor, 0)
            observations.append(stamp)
        return dict(features=torch.cat(features, dim=1), valid=torch.cat(validity, dim=1),
                    time=torch.cat(observations, dim=1))


@MODELS.register_module()
class TemporalHydroStateModel(HydroStateModel):
    """Share one encoder and decoder; condition each task group before decoding."""

    def __init__(self, query_time=None, **kwargs):
        super().__init__(**kwargs)
        groups = [tuple(group) for group in self.fraction_groups]
        grouped = {task for group in groups for task in group}
        self.query_groups = groups + [(task,) for task in self.tasks if task not in grouped]
        query_config = dict(query_time or {})
        query_config.setdefault("type", "TargetTimeQuery")
        self.query_time = MODELS.build(dict(
            query_config, channels=self.encoder.out_channels,
            tasks=[group[0] for group in self.query_groups]))

    def predict(self, inputs, data_samples=None):
        predictions = super().predict(inputs, data_samples)
        for index, prediction in enumerate(predictions):
            prediction["target_time"] = {
                name: bounds[index] for name, bounds in inputs["target_time"].items()}
        return predictions

    def _forward(self, inputs):
        if "target_time" not in inputs or set(inputs["target_time"]) != set(self.tasks):
            raise ValueError("Temporal model requires explicit target_time bounds for every task")
        sequence = self.encoder(inputs)
        keys = self.query_time.keys(sequence["features"].permute(0, 1, 3, 4, 2))
        outputs = {}
        for group in self.query_groups:
            bounds = inputs["target_time"][group[0]]
            for name in group[1:]:
                torch._assert_async((inputs["target_time"][name] == bounds).all(),
                                    "Joint cover fractions must share a target time")
            features = self.query_time(sequence, bounds, group[0], keys)
            if self.input_fusion is not None:
                features = self.input_fusion(features, inputs)
            decoded = self.decoder(features)
            if decoded.shape[-1] % self.output_scale or decoded.shape[-2] % self.output_scale:
                raise ValueError("Decoded grid must be divisible by output_scale")
            decoded = F.avg_pool2d(decoded, self.output_scale)
            outputs.update({task: self.heads[task](decoded) for task in group})
        return self._constrain_outputs(outputs)
