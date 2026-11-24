_base_ = ["./hydrostate_joint_d2moe.py"]

model = dict(type="NexusD2MoE", alignment_weight=0.1)
