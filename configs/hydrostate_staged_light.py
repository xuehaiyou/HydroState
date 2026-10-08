"""Half-width staged decoder, with the full sigmoid interaction architecture."""

_base_ = ["./hydrostate_staged.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_staged_light}}"

model = dict(
    decoder=dict(channels=[256, 128, 64]),
    # Auxiliary heads copy these final-head configurations automatically.
    heads=dict(
        water=dict(in_channels=64),
        soil_moisture=dict(in_channels=64),
        et=dict(in_channels=64),
    ),
    interaction=dict(channels=64),
)
