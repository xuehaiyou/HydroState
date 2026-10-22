"""Stage A: OlmoEarth shared/private LoRA; independent time queries and decoders."""

_base_ = ["./hydrostate_multilevel_temporal_l2.py"]

tasks = dict(
    water=dict(support="pixel", grid="input", station_index=0),
    soil_moisture=dict(support="window_mean", grid="scalar", station_index=1),
    et=dict(support="pml_native", grid="native", station_index=2),
)
statistics = "{{$HYDROSTATE_TARGET_STATS:__sp__}}"
if statistics == "__sp__":
    statistics = _base_.manifest.rsplit("/", 1)[0] + "/target_stats_shared_private_l2_v1.json"

model = dict(
    _delete_=True,
    type="SharedPrivateHydroStateModel",
    tasks=tasks,
    data_preprocessor=_base_.model.data_preprocessor,
    encoder=dict(
        **{k: v for k, v in _base_.model.encoder.items() if k != "type"},
        type="SharedPrivateOlmoEarthEncoder",
        shared_rank=8, private_rank=8, target_modules=["q", "v"],
        gate_init=0.5, lora_scale=1.0,
    ),
    decoder=_base_.model.decoder,
    heads=_base_.model.heads,
    query_time=dict(type="TargetTimeQuery", query_channels=64),
    loss=_base_.model.loss,
    target_statistics=statistics,
    train_shared=True,
    train_tasks=None,
    adapter_init=None,
    # WF and SM labels can have different dates. Only enable the opt-in
    # observed-WF support rule for explicitly aligned queries and labels.
    water_soil_coupling=None,
)

optim_wrapper = dict(
    _delete_=True,
    type="mmengine.AmpOptimWrapper", dtype="bfloat16", accumulative_counts=8,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    constructor="SharedPrivateOptimWrapperConstructor",
    paramwise_cfg=dict(shared_lr_mult=1.0, private_lr_mult=1.0, gate_lr_mult=1.0),
)
model_wrapper_cfg = dict(type="MMDistributedDataParallel", find_unused_parameters=True)
train_cfg = dict(max_epochs=30)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=30, T_max=30,
         eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/shared_private_joint}}"
load_from = None
resume = False
