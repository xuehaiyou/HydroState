"""E: fresh shared-LoRA task training with end-to-end AdapterFusion.

OlmoEarth retains pretrained initialization; no A/D task checkpoint is loaded.
"""

_base_ = ["./hydrostate_joint_shared.py"]

model = dict(
    type="AdapterFusionHydroStateModel",
    baseline_checkpoint=None,
    freeze_baseline=False,
    fusion_enabled=True,
)
train_cfg = dict(max_epochs=25)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=25, T_max=25, eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/adapterfusion_joint_scratch}}"
