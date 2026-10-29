"""Read several OlmoEarth depths in one masked encoder pass.

Uses the official v1/v1.2 token ordering, patch embedding, positional encoding,
attention blocks and token restoration. No hooks retain graphs across forwards
or checkpoint recomputations. The legacy single-output wrapper is unchanged.
"""

from __future__ import annotations

import torch

from hydrostate.registry import MODELS

from .olmoearth_encoder import OlmoEarthEncoder


def pool_modalities(tokens, masks):
    """Mean over valid times/bandsets, then over available modalities per pixel."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue

    pooled, available = [], []
    for name, value in tokens.items():
        valid = masks[f"{name}_mask"] == MaskValue.ONLINE_ENCODER.value
        count = valid.sum(dim=(3, 4))
        # FP32 accumulation; missing values must not contribute, even if NaN.
        value = value.float().masked_fill(~valid[..., None], 0)
        pooled.append(value.sum(dim=(3, 4)) / count.clamp(min=1)[..., None])
        available.append(count > 0)
    denominator = torch.stack(available).sum(0).clamp(min=1)
    result = torch.stack(pooled).sum(0) / denominator[..., None]
    return result.permute(0, 3, 1, 2).contiguous()


def multilevel_features(core, sample, patch_size, out_indices, input_res=10,
                        retain_tokens=False, branch_tasks=None):
    """Return normalized BCHW features, ordered shallow to deep.

    Always retain a key-validity mask, also in eval, so padded observations
    cannot change predictions when a sample is batched with longer sequences.
    The external packed flash-attn path is deliberately rejected; PyTorch SDPA
    (including its fused GPU kernels) remains supported.
    """
    if core.use_flash_attn:
        raise ValueError("Multilevel encoder requires use_flash_attn=False (PyTorch SDPA)")
    if (not out_indices or tuple(sorted(set(out_indices))) != tuple(out_indices)
            or out_indices[0] < 0 or out_indices[-1] >= len(core.blocks)):
        raise ValueError("out_indices must be unique increasing zero-based encoder block indices")
    if out_indices[-1] != len(core.blocks) - 1:
        raise ValueError("out_indices must include the last block to train the whole encoder")
    patchified = core.patch_embeddings.forward(sample, patch_size)
    values, masks, shapes = core.split_tokens_masks_and_dims(patchified)
    positions = core.build_rope_positions(
        values, masks, patch_size, input_res, timestamps=sample.timestamps,
    )
    encoded = core.composite_encodings(values, sample.timestamps, patch_size, input_res)
    encoded.update(masks)
    tokens, mask = core.collapse_and_combine_hwtc(encoded)
    tokens, indices, valid, _, _, original_valid = core._maybe_remove_masked_tokens(
        tokens, mask, fast_pass=False,
    )
    if positions is not None:
        positions, _, _, _, _ = core.remove_masked_tokens(positions, original_valid)
    attention_mask = valid
    if core.has_register_tokens:
        tokens, attention_mask = core.add_register_tokens_and_masks(tokens, attention_mask)
        if positions is not None:
            positions = core.add_register_positions(positions)
    features = [] if branch_tasks is None else {task: [] for task in branch_tasks}
    for index, block in enumerate(core.blocks):
        result = block(
            x=tokens, cu_seqlens=None, max_seqlen=None,
            attn_mask=attention_mask, rope_positions=positions,
        )
        if branch_tasks is not None and index in out_indices:
            tokens, *branches = result
            if len(branches) != len(branch_tasks):
                raise ValueError("Stage outputs disagree with configured tasks")
        else:
            tokens = result
        if index not in out_indices:
            continue
        selected = [(None, tokens)] if branch_tasks is None else zip(branch_tasks, branches)
        # Shared-LoRA exits reference the same tensor. Restore it once per depth;
        # private exits retain separate tensors and gradients.
        restored_features = {}
        for task, value in selected:
            key = id(value)
            if key not in restored_features:
                spatial = core.pop_register_tokens(value)[0] if core.has_register_tokens else value
                restored, _ = core.add_removed_tokens(core.norm(spatial), indices, valid)
                modality_tokens = core.split_and_expand_per_modality(restored, shapes)
                restored_features[key] = ((modality_tokens, masks) if retain_tokens else
                                          pool_modalities(modality_tokens, masks).to(tokens.dtype))
            feature = restored_features[key]
            (features if task is None else features[task]).append(feature)
    return (tuple(features) if branch_tasks is None else
            {task: tuple(values) for task, values in features.items()})


@MODELS.register_module()
class MultiLevelOlmoEarthEncoder(OlmoEarthEncoder):
    """OlmoEarth with intermediate outputs and unchanged pretrained parameter paths."""

    def __init__(self, out_indices=(3, 7, 11), **kwargs):
        super().__init__(**kwargs)
        self.out_indices = tuple(out_indices)
        core = self.backbone.model
        if (not self.out_indices or tuple(sorted(set(self.out_indices))) != self.out_indices
                or self.out_indices[0] < 0 or self.out_indices[-1] != len(core.blocks) - 1):
            raise ValueError("out_indices must increase and end at the final encoder block")
        if core.use_flash_attn:
            raise ValueError("Multilevel encoder requires the PyTorch SDPA attention backend")
        # This projection belongs to the pretraining instance-level objective;
        # our spatial retrieval forward never executes it. Keep original flags
        # for all other parameters, including fixed temporal/position encodings.
        core.project_and_aggregate.requires_grad_(False)

    def forward(self, inputs):
        context = self._make_context(inputs)
        if self.backbone.normalize:
            for sample_inputs in context.inputs:
                self.backbone.normalizer(sample_inputs, {})
        sample, _, _ = self.backbone._prepare_modality_inputs(context)
        features = multilevel_features(
            self.backbone.model, sample, self.patch_size, self.out_indices,
        )
        if any(value.shape[1] != self.out_channels for value in features):
            raise ValueError("Encoder feature channels do not match embedding_channels")
        return features
