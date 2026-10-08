"""20,000-sample NLL experiment; uses the current HPC model implementation."""
_base_ = ["./hydrostate_hpc.py"]

train_dataloader = dict(batch_size=16)
optim_wrapper = dict(accumulative_counts=1)
load_from = None
resume = False
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=125),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50,
         T_max=50, eta_min=1e-6),
]
custom_hooks = [dict(type="HydroVisualizationHook", num_samples=4, interval=1,
                     out_dir="visualizations", et_vmax=10.0)]
