"""Independent-LoRA control, retaining the same private prediction branches."""

_base_ = ["./hydrostate_shared_private.py"]
model = dict(encoder=dict(shared_rank=0, private_rank=16), train_shared=False)
train_cfg = dict(max_epochs=50)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50,
         eta_min_ratio=0.01),
]
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/private_lora_independent}}"
