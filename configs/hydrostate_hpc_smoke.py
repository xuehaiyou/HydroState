"""Eight optimizer updates to check the actual HPC training path."""

_base_ = ["./hydrostate_hpc.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_hpc_smoke}}"
train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=8)
train_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
optim_wrapper = dict(accumulative_counts=1)
param_scheduler = []
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
default_hooks = dict(checkpoint=None, logger=dict(interval=1))
log_processor = dict(by_epoch=False, window_size=1)
