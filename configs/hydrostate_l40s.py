"""L40S training: batch size 16 per GPU, effective batch size 16 x GPU count."""

_base_ = ["./hydrostate_hpc.py"]

train_dataloader = dict(batch_size=16)
optim_wrapper = dict(accumulative_counts=1)

custom_hooks = [
    dict(
        type="HydroVisualizationHook",
        num_samples=int("{{$HYDROSTATE_VIS_NUM_SAMPLES:4}}"),
        interval=1,
        out_dir="visualizations",
        et_vmax=10.0,
    ),
]

# Preserve the previous warmup of 1000 microsteps / 8 accumulation steps,
# now that every training iteration performs an optimizer update.
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=125),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        begin=0,
        end=50,
        T_max=50,
        eta_min=1e-6,
    ),
]
