"""MTLoRA product + station training; v3 split and training-only scales are mandatory."""

_base_ = ["./hydrostate_mtlora.py"]
site_split_root = "{{$HYDROSTATE_SITE_SPLIT_ROOT:__required_site_split_root__}}"
site_statistics = f"{site_split_root}/site_target_stats.json"
model = dict(
    site_supervision=dict(statistics=site_statistics, loss_weights=dict(site_sm=2.0, site_et=2.0))
)
train_dataloader = dict(
    sampler=dict(type="HydroBalancedSampler", shuffle=True),
    dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root, site_fraction=0.5),
)
val_dataloader = dict(dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root))
test_dataloader = dict(dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root))
val_evaluator = dict(
    type="JointHydroStateMetric", site_statistics=site_statistics, site_score_weight=0.5
)
test_evaluator = dict(
    type="JointHydroStateMetric", site_statistics=site_statistics, site_score_weight=0.5
)
default_hooks = dict(checkpoint=dict(save_best="joint_score"))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/retrieval_joint}}"
