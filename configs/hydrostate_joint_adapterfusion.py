"""D: frozen joint shared-LoRA baseline plus official attention composition."""

_base_ = ["./hydrostate_joint_shared.py"]

model = dict(
    type="AdapterFusionHydroStateModel",
    baseline_checkpoint="{{$HYDROSTATE_BASELINE_CHECKPOINT:__required_baseline_checkpoint__}}",
    fusion_enabled=True,
)
optim_wrapper = dict(accumulative_counts=1)
train_cfg = dict(max_epochs=10)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=100),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=10, T_max=10, eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/adapterfusion_joint}}"
