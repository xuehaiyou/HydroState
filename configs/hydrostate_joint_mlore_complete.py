"""Complete official ViT-B MLoRE decoder on four OlmoEarth temporal depth queries."""

_base_ = ['./hydrostate_joint.py']
model = dict(
    type='GroupedExpertHydroStateModel',
    branches=None,
    router=None,
    official_heads=True,
    encoder=dict(out_indices=[2, 5, 8, 11], num_levels=4),
    decoder=dict(_delete_=True, type='CompleteMLoREDecoder',
                 groups=['coverage', 'sm', 'et'], in_channels=768, channels=640,
                 grid_size=30, ranks=tuple(range(16, 129, 8)), private_rank=64,
                 topk=9, pre_softmax=False, balance_weight=.0001, quality_channels=32),
)
work_dir = '{{$HYDROSTATE_WORK_DIR:work_dirs/mlore_complete_joint}}'
