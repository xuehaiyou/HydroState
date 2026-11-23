_base_ = ["./hydrostate_joint_independent.py"]

model = dict(encoder=dict(_delete_=True, type="AnySatEncoder", input_profile=_base_.input_profile,
                          source="{{$HYDROSTATE_ANYSAT_SOURCE:/scr/u/xiaoz/HydroState/pretrained/AnySat/src}}",
                          weights="{{$HYDROSTATE_ANYSAT_WEIGHTS:/scr/u/xiaoz/HydroState/pretrained/AnySat/AnySat.pth}}"))
