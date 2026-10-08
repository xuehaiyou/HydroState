"""Eight real L40S updates and full validation before staged Sigmoid training."""

_base_ = ["./hydrostate_staged.py"]

custom_imports = dict(
    imports=["hydrostate", "tools.validation.nll_hook"], allow_failed_imports=False
)
custom_hooks = _base_.custom_hooks + [dict(type="NLLVerificationHook")]
train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=8, val_interval=8)
train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(num_workers=0, persistent_workers=False)
param_scheduler = []
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8, save_best=None), logger=dict(interval=1)
)
log_processor = dict(by_epoch=False, window_size=1)
