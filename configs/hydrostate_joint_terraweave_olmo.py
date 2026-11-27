_base_ = ["./hydrostate_joint_terraweave.py"]

model = dict(sources=("olmoearth",), anysat=None, decoder=dict(sources=("shared",)))
