"""Stage B: select one task and load a stage-A checkpoint; keep knowledge fixed."""

_base_ = ["./hydrostate_shared_private.py"]
specialize_task = "{{$HYDROSTATE_SPECIALIZE_TASK:soil_moisture}}"
model = dict(train_shared=False, train_tasks=[specialize_task])
load_from = "{{$HYDROSTATE_SP_CHECKPOINT:__SET_HYDROSTATE_SP_CHECKPOINT__}}"
resume = False  # A new optimizer/schedule, not a stage-A optimizer resume.
train_cfg = dict(max_epochs=20)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=20, T_max=20,
         eta_min_ratio=0.01),
]
default_hooks = dict(checkpoint=dict(save_best=f"{specialize_task}_rmse", rule="less"))
work_dir = "{{$HYDROSTATE_WORK_DIR:__task__}}"
if work_dir == "__task__":
    work_dir = f"work_dirs/shared_private_specialize_{specialize_task}"
