"""Three-variable, temporal MTLoRA adaptation with pretrained-compatible L1 input.

Logical stages end at blocks 4/8/12; these are depths, not spatial scales.
Use newly generated temporal-contract training statistics, not L2 statistics.
"""

_base_ = ["./hydrostate_hpc.py"]

input_profile = "olmoearth_l1_v1"
input_packs = []
target_time_queries = True
statistics = "{{$HYDROSTATE_TARGET_STATS:__mtlora__}}"
if statistics == "__mtlora__":
    statistics = _base_.manifest.rsplit("/", 1)[0] + "/target_stats_mtlora_temporal_v1.json"

model = dict(
    type="MTLoRAHydroStateModel",
    encoder=dict(
        type="MTLoRAOlmoEarthEncoder",
        out_indices=[3, 7, 11],
        input_profile=input_profile,
        shared_rank=8,
        private_rank=4,
        lora_scale=1.0,
        train_patch_embedding=True,
        train_layer_norm=True,
    ),
    decoder=dict(
        _delete_=True,
        type="MTLoRAFusionDecoder",
        in_channels=768,
        channels=64,
        num_levels=3,
        scale_factor=4,
    ),
    heads=dict(
        water=dict(in_channels=64), soil_moisture=dict(in_channels=64), et=dict(in_channels=64)
    ),
    query_time=dict(type="TargetTimeQuery", query_channels=64),
    target_statistics=statistics,
    water_soil_coupling=dict(
        enabled=True,
        supervision_source="observed",
        min_land_fraction=0.01,
        min_window_land_fraction=0.01,
    ),
    land_time_policy="aligned_or_window",
)
optim_wrapper = dict(
    _delete_=True,
    type="mmengine.AmpOptimWrapper",
    dtype="bfloat16",
    accumulative_counts=8,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50, eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/mtlora}}"
load_from = None
resume = False
