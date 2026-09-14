_base_ = ["./_base_/dataset.py", "./_base_/model.py", "./_base_/runtime.py"]

custom_imports = dict(imports=["hydrostate"], allow_failed_imports=False)

work_dir = "/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_v1"

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _scope_="mmengine",
    type="mmengine.AmpOptimWrapper",
    dtype="bfloat16",
    accumulative_counts=4,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(type="mmengine.AdamW", lr=1e-4, weight_decay=0.05),
    constructor="mmengine.OlmoEarthLayerDecayOptimWrapperConstructor",
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        begin=0,
        end=50,
        T_max=50,
        eta_min=1e-6,
    ),
]
