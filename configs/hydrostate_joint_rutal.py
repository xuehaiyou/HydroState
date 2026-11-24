_base_ = ["./hydrostate_joint_independent.py"]

model = dict(
    type="FoundationStageModelV2",
    encoder=dict(type="FoundationStageEncoderV2", method="rutal", gradient_checkpointing=True,
                 out_indices=[2, 5, 8, 11], num_levels=4),
    decoder=dict(type="RUTaLDecoder", channels=768, num_levels=5),
)
