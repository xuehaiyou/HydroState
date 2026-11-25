_base_ = ["./hydrostate_joint_multifm_rutal.py"]

model = dict(decoder=dict(learn_source_weights=False))
