_base_ = ["./hydrostate_joint_rutal.py"]

model = dict(
    type="MultiFMRUTaL",
    anysat=dict(
        type="AnySatEncoder",
        source="{{$HYDROSTATE_ANYSAT_SOURCE:/scr/u/xiaoz/HydroState/pretrained/AnySat/src}}",
        weights="{{$HYDROSTATE_ANYSAT_WEIGHTS:/scr/u/xiaoz/HydroState/pretrained/AnySat/AnySat.pth}}",
        input_profile=_base_.input_profile,
        ladder_features=True,
    ),
    decoder=dict(type="MultiFMRUTaLDecoder", learn_source_weights=True),
)
