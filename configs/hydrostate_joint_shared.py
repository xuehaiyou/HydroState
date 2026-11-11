"""Fresh shared-LoRA with the same joint supervision as CoSA B/C."""

_base_ = ["./hydrostate_mtlora_shared.py"]

site_split_root = "{{$HYDROSTATE_SITE_SPLIT_ROOT:__required_site_split_root__}}"
site_statistics = f"{site_split_root}/site_target_stats.json"
model = dict(
    encoder=dict(gradient_checkpointing=False),
    site_supervision=dict(statistics=site_statistics, loss_weights=dict(site_sm=2.0, site_et=2.0)),
)
train_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=True, round_up=False),
    dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root),
)
val_dataloader = dict(dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root))
test_dataloader = dict(dataset=dict(type="JointRetrievalDataset", site_split_root=site_split_root))
val_evaluator = dict(
    type="JointHydroStateMetric", site_statistics=site_statistics, site_score_weight=2 / 3,
    comparison_years=(2017, 2024),
)
test_evaluator = dict(
    type="JointHydroStateMetric", site_statistics=site_statistics, site_score_weight=2 / 3,
    comparison_years=(2017, 2024),
)
default_hooks = dict(checkpoint=dict(save_best="joint_score"))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/shared_lora_joint}}"
