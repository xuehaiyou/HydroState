"""Experiment C: fresh shared-LoRA + CoSA with conditional Q/K routing."""

_base_ = ["./_base_/retrieval.py"]

model = dict(
    type="CoSAHydroStateModel",
    encoder=dict(
        type="CoSASharedLoRAEncoder",
        model_id="OLMOEARTH_V1_2_BASE",
        model_path="{{$HYDROSTATE_MODEL_PATH:/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}}",
        input_profile=_base_.input_profile,
        out_indices=[3, 7, 11],
        patch_size=4,
        embedding_channels=768,
        inputs_are_normalized=True,
        shared_rank=8,
        private_rank=0,
        lora_scale=1.0,
        use_landsat_toa_adapter=False,
        train_patch_embedding=True,
        train_layer_norm=True,
        gradient_checkpointing=False,
    ),
    decoder=dict(type="CoSAStateFusion", in_channels=768, channels=256,
                 num_levels=3, num_heads=4, quality_channels=32),
    expert_groups=dict(
        coverage=["water", "snow_ice", "land"],
        sm=["soil_moisture"],
        et=["et_water", "et_ice", "et_land"],
    ),
    branches=dict(channels=64),
    router=dict(enabled=True, task_embedding_dim=32, routing_dim=64,
                residual_init=0.0, detach_source=True),
)
# Both variants inherit the same seed and runtime as the shared-LoRA baseline.
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/cosa}}"
