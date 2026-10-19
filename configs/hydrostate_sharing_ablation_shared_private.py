"""B: rank 8 shared + rank 8 private, coefficient 1, 50 joint epochs."""

_base_ = ["./hydrostate_shared_private_joint50.py"]

ablation_seed = int("{{$HYDROSTATE_ABLATION_SEED:3407}}")
randomness = dict(seed=ablation_seed)
model = dict(
    encoder=dict(shared_gate_mode="fixed_one", shared_rank=8, private_rank=8),
    branch_init_seed=ablation_seed,
    train_shared=True, train_tasks=None, active_tasks_only=False,
)
# BF16 needs no loss scaling. With only one supervised variable an accumulation
# window can have no labels/owned gradients; GradScaler.step would assert then.
optim_wrapper = dict(loss_scale=dict(enabled=False))
# Retain epoch 50 plus one validation-selected checkpoint; bound scratch usage.
default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_last=True,
                                    save_best="hydro_score", rule="greater"))
# Comparable logging without task-specific visualization overhead.
custom_hooks = []
work_dir = "{{$HYDROSTATE_WORK_DIR:__ablation__}}"
if work_dir == "__ablation__":
    work_dir = f"work_dirs/sharing_ablation/shared_private_seed{ablation_seed}"
