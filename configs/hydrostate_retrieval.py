"""Current five-slot retrieval: MTLoRA, separate ET branches, weak SM supervision."""

import importlib

_contract = importlib.import_module("hydrostate.data.retrieval.contract")
_specs = importlib.import_module("hydrostate.data.retrieval.specs")

_base_ = ["./_base_/runtime.py"]
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
    type="MTLoRAHydroStateModel",
    tasks=tasks,
    output_scale=_contract.OUTPUT_SCALE,
    data_preprocessor=dict(type="HydroDataPreprocessor", input_profile=input_profile),
    encoder=dict(
        type="MTLoRAOlmoEarthEncoder",
        model_id="OLMOEARTH_V1_2_BASE",
        model_path="{{$HYDROSTATE_MODEL_PATH:/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}}",
        input_profile=input_profile,
        out_indices=[3, 7, 11],
        patch_size=4,
        embedding_channels=768,
        inputs_are_normalized=True,
        shared_rank=8,
        private_rank=4,
        lora_scale=1.0,
        train_patch_embedding=True,
        train_layer_norm=True,
        gradient_checkpointing=True,
    ),
    decoder=dict(
        type="MTLoRAFusionDecoder", in_channels=768, channels=64, num_levels=3, scale_factor=4
    ),
    heads={
        name: dict(
            type="RegressionHead",
            in_channels=64,
            activation=_specs.LABEL_SPECS[name].activation,
        )
        for name in tasks
    },
    fraction_groups=[["water", "snow_ice", "land"]],
    query_time=dict(type="TargetTimeQuery", query_channels=64),
    input_fusion=dict(type="CompositeQualityFusion", feature_channels=768),
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
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/retrieval_mtlora}}"
