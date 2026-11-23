_base_ = ["./hydrostate_joint_independent.py"]

model = dict(
    type="FoundationStageModelV2",
    encoder=dict(type="FoundationStageEncoderV2", method="ditask", gradient_checkpointing=True,
                 out_indices=[1, 3, 9, 11], num_levels=4),
    decoder=dict(type="DiTASKStateFusion", num_levels=4),
)
