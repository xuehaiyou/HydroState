"""A: one rank 16 LoRA shared by all tasks, with identical private readouts."""

_base_ = ["./hydrostate_sharing_ablation_shared_private.py"]
model = dict(encoder=dict(shared_rank=16, private_rank=0))
work_dir = "{{$HYDROSTATE_WORK_DIR:__ablation__}}"
if work_dir == "__ablation__":
    work_dir = f"work_dirs/sharing_ablation/shared_seed{_base_.ablation_seed}"
