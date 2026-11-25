_base_ = ["./hydrostate_joint_multifm_bdta.py"]

model = dict(decoder=dict(learn_source_weights=False))
