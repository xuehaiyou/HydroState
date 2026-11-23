_base_ = ["./hydrostate_joint_independent.py"]

model = dict(
    type="FoundationStageModelV2",
    encoder=dict(type="FoundationStageEncoderV2", method="d2moe", gradient_checkpointing=True,
                 out_indices=[2, 5, 8, 11], num_levels=4),
    decoder=dict(type="D2MoEStateFusion", channels=384, num_levels=4),
)
