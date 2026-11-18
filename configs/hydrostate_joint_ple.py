"""Two-level PLE with all intermediate shared/task gates (paper §4.2)."""

_base_ = ['./hydrostate_joint.py']
model = dict(
    type='GroupedExpertHydroStateModel',
    branches=None,
    router=None,
    decoder=dict(_delete_=True, type='PLEDecoder', groups=['coverage', 'sm', 'et'],
                 temporal=dict(in_channels=768, channels=256, num_levels=3,
                               num_heads=4, quality_channels=32),
                 widths=(256, 128), num_shared=1, num_private=1, tower_channels=64),
)
work_dir = '{{$HYDROSTATE_WORK_DIR:work_dirs/ple_joint}}'
