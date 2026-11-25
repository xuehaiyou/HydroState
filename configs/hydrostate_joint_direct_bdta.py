_base_ = ["./hydrostate_joint_multifm_bdta.py"]

model = dict(type="DirectMultiFMBDTA", decoder=dict(type="DirectBDTADecoder"))
