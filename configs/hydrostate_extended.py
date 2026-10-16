"""Example: existing hydro tasks plus a temperature label and terrain input.

Create the two packs first; paths are relative to each dataset's data_root.
The CLI propagates the shared task/pack selections to datasets and evaluators.
"""

_base_ = ["./hydrostate_v1.py"]

tasks = dict(
    water=dict(support="pixel", grid="input", station_index=0),
    soil_moisture=dict(support="window_mean", grid="scalar", station_index=1),
    et=dict(support="pml_native", grid="native", station_index=2),
    temperature=dict(support="pixel", grid="output"),
)
input_packs = [dict(path="inputs/terrain_v1", variables=["elevation"])]
label_packs = [dict(path="labels/temperature_v1", variables=["temperature"])]

model = dict(
    heads=dict(temperature=dict(type="RegressionHead", in_channels=128, activation="identity")),
    input_fusion=dict(
        type="AuxiliaryInputFusion", feature_channels=768,
        # Metres / 1000 is an explicit fixed physical scaling, not fitted statistics.
        inputs=dict(elevation=dict(channels=1, mean=[0.0], std=[1000.0], temporal="static")),
    ),
    target_statistics="{{$HYDROSTATE_TARGET_STATS:__manifest__}}",
)
work_dir = "/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_extended"
