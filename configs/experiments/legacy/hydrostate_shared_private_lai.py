"""Incremental LAI: import old branches/shared knowledge, train only LAI."""

_base_ = ["./hydrostate_shared_private.py"]
tasks = dict(**_base_.tasks, lai=dict(support="native", grid="native"))
label_packs = [dict(path="{{$HYDROSTATE_LAI_PACK:labels/ecosystem_v1}}", variables=["lai"])]
statistics = "{{$HYDROSTATE_TARGET_STATS:__lai__}}"
if statistics == "__lai__":
    statistics = _base_.manifest.rsplit("/", 1)[0] + "/target_stats_shared_private_lai_v1.json"
model = dict(
    tasks=tasks, train_shared=False, train_tasks=["lai"], target_statistics=statistics,
    heads=dict(lai=dict(type="RegressionHead", in_channels=64,
                        hidden_channels=32, activation="softplus")),
    adapter_init=dict(_delete_=True, path="{{$HYDROSTATE_SP_BUNDLE:__SET_HYDROSTATE_SP_BUNDLE__}}"),
)
default_hooks = dict(checkpoint=dict(save_best="lai_rmse", rule="less"))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/shared_private_lai}}"
