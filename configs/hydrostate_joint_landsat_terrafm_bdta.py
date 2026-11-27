_base_ = ["./hydrostate_joint_bdta.py"]

model = dict(
    type="LandsatTerraBDTA",
    encoder=dict(
        _delete_=True,
        type="LandsatTerraEncoder",
        input_profile=_base_.input_profile,
        weights_root="{{$HYDROSTATE_SSL4EO_TERRAFM:pretrained/ssl4eo_terrafm}}",
    ),
    decoder=dict(type="LandsatTerraBDTADecoder", learn_source_weights=True),
)
