"""Multilevel OlmoEarth + shared light decoder + output-grid task adapters.

Uses the existing window dataset: SM supervision is still a window-center
approximation, NOT full SMAP footprint supervision. See MODEL_ARCHITECTURES.md.
"""

_base_ = ["./hydrostate_l40s.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_multilevel}}"
load_from = None
resume = False

model = dict(
    type="MultiLevelHydroStateModel",
    encoder=dict(type="MultiLevelOlmoEarthEncoder", out_indices=[3, 7, 11]),
    decoder=dict(
        _delete_=True, type="MultiLevelFusionDecoder", in_channels=768, channels=[256, 128, 64],
    ),
    adapter=dict(enabled=True, bottleneck_channels=16),
    interaction=dict(
        type="TaskGatedInteraction", channels=64, mode="sigmoid",
        residual_scale=0.1, learnable_scale=True, merge_ops=True,
    ),
    heads=dict(
        water=dict(in_channels=64, hidden_channels=32),
        soil_moisture=dict(in_channels=64, hidden_channels=32),
        et=dict(in_channels=64, hidden_channels=32),
    ),
    water_soil_coupling=dict(
        enabled=True, supervision_source="observed",
        min_land_fraction=0.01, min_window_land_fraction=0.01,
    ),
)

optim_wrapper = dict(paramwise_cfg=dict(recipe="official"))
# A relative cosine floor preserves LLRD: an absolute 1e-6 floor would exceed
# the input embedding's 5.69e-7 base LR and increase its LR late in training.
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=125),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50,
         eta_min_ratio=0.01),
]
