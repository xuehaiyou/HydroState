"""MTLoRA architecture on the shared product retrieval configuration."""

_base_ = ["./_base_/retrieval.py"]

model = dict(
    type="MTLoRAHydroStateModel",
    encoder=dict(
        type="MTLoRAOlmoEarthEncoder",
        model_id="OLMOEARTH_V1_2_BASE",
        model_path="{{$HYDROSTATE_MODEL_PATH:/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}}",
        input_profile=_base_.input_profile,
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
    query_time=dict(type="TargetTimeQuery", query_channels=64),
    input_fusion=dict(type="CompositeQualityFusion", feature_channels=768),
)
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/retrieval_mtlora}}"
