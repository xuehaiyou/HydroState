"""Incremental instantaneous LST; input bands remain the existing SR profile."""

_base_ = ["hydrostate_shared_private.py"]
tasks = dict(**_base_.tasks, lst=dict(support="pixel", grid="output"))
label_packs = [dict(path="{{$HYDROSTATE_LST_PACK:labels/lst_observation_v1}}",
                    variables=["lst"], max_uncertainty=dict(lst=2.0))]
statistics = "{{$HYDROSTATE_TARGET_STATS:__lst__}}"
if statistics == "__lst__":
    statistics = _base_.manifest.rsplit("/", 1)[0] + "/target_stats_shared_private_lst_v1.json"
model = dict(
    tasks=tasks, train_shared=False, train_tasks=["lst"], target_statistics=statistics,
    heads=dict(lst=dict(type="RegressionHead", in_channels=64,
                        hidden_channels=32, activation="identity")),
    adapter_init=dict(_delete_=True, path="{{$HYDROSTATE_SP_BUNDLE:__SET_HYDROSTATE_SP_BUNDLE__}}"),
)
default_hooks = dict(checkpoint=dict(save_best="lst_rmse", rule="less"))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/shared_private_lst}}"
