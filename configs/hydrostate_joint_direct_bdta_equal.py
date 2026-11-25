_base_ = ["./hydrostate_joint_direct_bdta.py"]

model = dict(decoder=dict(learn_source_weights=False))
