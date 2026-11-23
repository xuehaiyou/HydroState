_base_ = ["./hydrostate_joint_independent.py"]

model = dict(type="FoundationStageModel", encoder=dict(type="FoundationStageEncoder",
             method="d2moe", gradient_checkpointing=True))
