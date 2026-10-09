"""Light Sigmoid model with land-conditional SM and WF-weighted supervision."""

_base_ = ["./hydrostate_staged_light.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_staged_light_land_sm}}"

model = dict(
    water_soil_coupling=dict(
        enabled=True,
        # Fixed observed WF avoids prediction-dependent validation support.
        # Optional "predicted" uses detached final WF for both SM heads.
        supervision_source="observed",
        min_land_fraction=0.01,
        min_window_land_fraction=0.01,
    ),
)
