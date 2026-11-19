_base_ = ['./hydrostate_joint.py']
model = dict(type='GroupedExpertHydroStateModel', branches=None, router=None,
             encoder=dict(out_indices=[2, 5, 8, 11], num_levels=4),
             decoder=dict(_delete_=True, type='TaskAttentionDecoder', method='mtan',
                          groups=['coverage', 'sm', 'et']))
work_dir = 'work_dirs/MTAN'
