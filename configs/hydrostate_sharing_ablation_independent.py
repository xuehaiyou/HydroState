"""C: one task per process/optimizer, rank 16 private and no shared adaptation."""

_base_ = ["./hydrostate_sharing_ablation_shared_private.py"]
independent_task = "{{$HYDROSTATE_ABLATION_TASK:soil_moisture}}"
if independent_task not in ("water", "soil_moisture", "et"):
    raise ValueError("Select water, soil_moisture or et for the independent control")
model = dict(
    encoder=dict(shared_rank=0, private_rank=16),
    train_shared=False, train_tasks=[independent_task], active_tasks_only=True,
)
# Keep the complete input/statistics contract; only this task is run/evaluated.
val_evaluator = dict(evaluated_tasks=[independent_task])
test_evaluator = dict(evaluated_tasks=[independent_task])
default_hooks = dict(checkpoint=dict(save_best=f"{independent_task}_rmse", rule="less"))
work_dir = "{{$HYDROSTATE_WORK_DIR:__ablation__}}"
if work_dir == "__ablation__":
    work_dir = (f"work_dirs/sharing_ablation/independent_{independent_task}"
                f"_seed{_base_.ablation_seed}")
