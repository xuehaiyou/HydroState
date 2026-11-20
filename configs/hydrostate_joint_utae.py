_base_ = ['./hydrostate_joint.py']
model = dict(_base_.model)
for key in ('expert_groups', 'branches', 'router'):
    model.pop(key)
model.update(_delete_=True, type='CentralWindowModel',
             encoder=dict(type='UTAEInput', input_profile=_base_.input_profile),
             decoder=dict(type='UTAEDecoder'))
work_dir = 'work_dirs/utae_joint'
