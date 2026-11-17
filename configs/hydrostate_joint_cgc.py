"""CGC screening candidate: same CoSA state, private experts and supervision."""

_base_ = ["./hydrostate_joint.py"]

model = dict(router=dict(_delete_=True, type="CGCRouter", num_shared=2))
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/cgc_joint}}"
