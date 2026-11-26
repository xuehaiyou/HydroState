_base_ = ["./hydrostate_joint_multifm_bdta.py"]

model = dict(
    type="TerraWeave",
    encoder=dict(method="terraweave", gradient_checkpointing=False),
    anysat=dict(frozen=True, ladder_features=True, include_patch=False),
    decoder=dict(
        _delete_=True,
        type="TerraWeaveDecoder",
        in_channels=768,
        channels=128,
        num_levels=4,
        num_heads=4,
        window_size=4,
        quality_channels=32,
    ),
)
