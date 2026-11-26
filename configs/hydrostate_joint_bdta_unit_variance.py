_base_ = ["./hydrostate_joint_bdta.py"]

model = dict(loss=dict(learn_variance=False))
