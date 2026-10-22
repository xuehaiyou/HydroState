"""Retrospective ecosystem retrieval with one actual-time LST label per sample."""

_base_ = ["hydrostate_ecosystem.py"]

ecosystem_pack = "{{$HYDROSTATE_LABEL_PACK:labels/ecosystem_v1}}"
lst_pack = "{{$HYDROSTATE_LST_PACK:labels/lst_observation_v1}}"
label_packs = [
    dict(path=ecosystem_pack, variables=["water", "trees", "grass", "flooded_vegetation",
         "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice", "lai", "gpp",
         "albedo_bsa", "albedo_wsa"]),
    dict(path=lst_pack, variables=["lst"], max_uncertainty=dict(lst=2.0)),
]
target_time_queries = True
# A batch/rank may have no observations for a satellite modality.
find_unused_parameters = True
data_root = "{{$HYDROSTATE_DATA_ROOT:/mnt/d/Hydrostate/samples/merged_v1}}"
statistics = "{{$HYDROSTATE_TARGET_STATS:__temporal__}}"
if statistics == "__temporal__":
    statistics = f"{data_root}/target_stats_ecosystem_temporal_v1.json"
model = dict(type="TemporalHydroStateModel",
             encoder=dict(type="TemporalOlmoEarthEncoder"),
             query_time=dict(type="TargetTimeQuery", query_channels=64), target_statistics=statistics)
work_dir = "work_dirs/hydrostate_ecosystem_temporal"
