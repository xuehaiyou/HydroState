_base_ = ["./hydrostate_joint_multifm_rutal.py"]

model = dict(
    type="MultiFMBDTA",
    encoder=dict(method="bdta", gradient_checkpointing=False),
    anysat=dict(frozen=True),
    decoder=dict(type="MultiFMBDTADecoder", learn_source_weights=True),
)
