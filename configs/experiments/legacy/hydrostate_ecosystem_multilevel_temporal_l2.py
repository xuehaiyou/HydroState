"""The existing sixteen tasks with the same L2 satellite replacement contract."""

_base_ = ["./hydrostate_ecosystem_multilevel_temporal_hpc.py"]

input_profile = "landsat_l2_sr_v1"
input_packs = [dict(path="{{$HYDROSTATE_L2_PACK:inputs/landsat_l2_sr_v1}}",
                    variables=["landsat_l2"])]
statistics = "{{$HYDROSTATE_TARGET_STATS:__ecosystem_l2__}}"
if statistics == "__ecosystem_l2__":
    statistics = _base_.manifest.rsplit("/", 1)[0] + "/target_stats_ecosystem_temporal_l2_v1.json"
model = dict(encoder=dict(input_profile=input_profile),
             data_preprocessor=dict(input_profile=input_profile), target_statistics=statistics)
work_dir = "{{$HYDROSTATE_WORK_DIR:work_dirs/ecosystem_multilevel_temporal_l2}}"
