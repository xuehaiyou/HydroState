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
    loss=dict(type="MaskedHuberLoss", delta=1.0),
    task_weights=dict(water=1.0, soil_moisture=0.25, et=1.0),
    # Physical reference scales, not measured dataset standard deviations.
    task_scales=dict(water=1.0, soil_moisture=0.1, et=3.0),
    site_loss_weight=3.0,
)

