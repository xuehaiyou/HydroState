"""Sixteen tasks: multilevel fusion + shared time queries + per-task adapters."""

_base_ = ["./hydrostate_ecosystem_temporal_hpc.py"]

work_dir = (
    "{{$HYDROSTATE_WORK_DIR:/scr/u/xiaoz/HydroState/work_dirs/ecosystem_multilevel_temporal}}"
)

model = dict(
    type="MultiLevelTemporalHydroStateModel",
    encoder=dict(type="MultiLevelTemporalOlmoEarthEncoder", out_indices=[3, 7, 11]),
    decoder=dict(
        _delete_=True, type="MultiLevelFusionDecoder", in_channels=768, channels=[256, 128, 64],
    ),
    adapter=dict(enabled=True, bottleneck_channels=16),
    interaction=dict(
        type="TaskGatedInteraction", channels=64, mode="sigmoid",
        residual_scale=0.1, learnable_scale=True, merge_ops=True,
    ),
    heads={name: dict(in_channels=64, hidden_channels=32) for name in _base_.model.heads},
    # Keep the existing 16-task supervision and statistics. WF and SM can query
    # different dates, so the WF output is not an aligned SM land mask.
    water_soil_coupling=None,
)

# Retain the multilevel model's official LLRD recipe and relative cosine floor.
# Batch 2 x accumulation 8 preserves effective batch 16; warm up for 125 updates.
optim_wrapper = dict(paramwise_cfg=dict(recipe="official"))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50,
         eta_min_ratio=0.01),
]
