default_scope = "hydrostate"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="hydro_score",
        rule="greater",
        max_keep_ckpts=3,
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(by_epoch=True, window_size=20)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=3407, deterministic=False)

