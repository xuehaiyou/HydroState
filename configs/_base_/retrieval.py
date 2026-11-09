"""Shared retrieval data, physical supervision and training budget."""

# Declare classic MMEngine inheritance before imports; stdlib detection differs in HPC envs.
_base_ = ["./runtime.py"]

import importlib  # noqa: E402 -- MMEngine inheritance mode must be declared first.

_contract = importlib.import_module("hydrostate.data.retrieval.contract")
_specs = importlib.import_module("hydrostate.data.retrieval.specs")

custom_imports = dict(
    imports=["hydrostate.models", "hydrostate.datasets", "hydrostate.evaluation"],
    allow_failed_imports=False,
)
data_root = "{{$HYDROSTATE_DATA_ROOT:/fossfs/xiaozhen/HydroState/retrieval_5x8_eight_groups}}"
manifest = "{{$HYDROSTATE_MANIFEST:__data_root__}}"
if manifest == "__data_root__":
    manifest = f"{data_root}/samples.parquet"
input_profile = _contract.PROFILE
tasks = _contract.TASKS
data_contract = _contract.VERSION
label_specs = dict()
label_packs = []

model = dict(
    tasks=tasks,
    output_scale=_contract.OUTPUT_SCALE,
    data_preprocessor=dict(type="HydroDataPreprocessor", input_profile=input_profile),
    heads={
        name: dict(
            type="RegressionHead",
            in_channels=64,
            activation=_specs.LABEL_SPECS[name].activation,
        )
        for name in tasks
    },
    fraction_groups=[["water", "snow_ice", "land"]],
    water_soil_coupling=dict(
        enabled=True,
        supervision_source="observed",
        ice_task="snow_ice",
        min_land_fraction=0.01,
        min_window_land_fraction=0.95,
        min_known_fraction=0.95,
    ),
    land_time_policy="require_aligned",
    task_loss_weights=dict(water=1 / 3, snow_ice=1 / 3, land=1 / 3, soil_moisture=0.25),
    loss=dict(type="UncertaintyWeightedNLL", min_variance=1e-6),
    target_statistics="{{$HYDROSTATE_TARGET_STATS:__manifest__}}",
)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="hydro_collate"),
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RetrievalDataset",
        manifest=manifest,
        data_root=data_root,
        split="train",
        training=True,
        tasks=tasks,
        input_profile=input_profile,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="hydro_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="RetrievalDataset",
        manifest=manifest,
        data_root=data_root,
        split="validation",
        tasks=tasks,
        input_profile=input_profile,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="hydro_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="RetrievalDataset",
        manifest=manifest,
        data_root=data_root,
        split="test",
        tasks=tasks,
        input_profile=input_profile,
    ),
)
val_evaluator = dict(type="HydroStateMetric")
test_evaluator = dict(type="HydroStateMetric")
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    accumulative_counts=8,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=50, T_max=50, eta_min_ratio=0.01),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
find_unused_parameters = True
