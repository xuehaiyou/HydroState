"""Ablation: shared/private jointly trained for the entire 50-epoch budget."""

_base_ = ["./hydrostate_shared_private.py"]
train_cfg = dict(max_epochs=50)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50,
         eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/shared_private_joint50}}"
