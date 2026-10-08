model = dict(
    type="HydroStateModel",
    data_preprocessor=dict(type="HydroDataPreprocessor"),
    encoder=dict(
        type="OlmoEarthEncoder",
        model_id="OLMOEARTH_V1_2_BASE",
        model_path="{{$HYDROSTATE_MODEL_PATH:/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base}}",
        gradient_checkpointing=True,
        patch_size=4,
        embedding_channels=768,
        # Zarr creation must apply the official normalization and record its version.
        inputs_are_normalized=True,
    ),
    decoder=dict(
        type="SharedUNetDecoder",
        in_channels=768,
        channels=[512, 256, 128],
        scale_factor=4,
    ),
    heads=dict(
        water=dict(type="RegressionHead", in_channels=128, activation="sigmoid"),
        soil_moisture=dict(type="RegressionHead", in_channels=128, activation="sigmoid"),
        et=dict(type="RegressionHead", in_channels=128, activation="softplus"),
    ),
    loss=dict(type="UncertaintyWeightedNLL", min_variance=1e-6),
    # The CLI resolves this to target_stats.json beside the training manifest,
    # verifies the training fingerprint and embeds its contents in the run config.
    target_statistics="{{$HYDROSTATE_TARGET_STATS:__manifest__}}",
)
