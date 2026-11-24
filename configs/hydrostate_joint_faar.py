_base_ = ["./hydrostate_joint_independent.py"]

model = dict(
    type="FoundationStageModelV2",
    encoder=dict(type="FoundationStageEncoderV2", method="faar", gradient_checkpointing=True,
                 out_indices=[2, 5, 8, 11], num_levels=4),
    decoder=dict(type="FAARDecoder", channels=256, num_levels=4),
)
custom_hooks = [dict(type="FAARRankHook")]
