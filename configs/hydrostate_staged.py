"""Progressive task branches, auxiliary MSE and one sigmoid-gated exchange.

Shares the L40S data/optimization recipe. Batch size 16 is inherited, not a
measured memory guarantee for this larger architecture; run a smoke job first.
"""

_base_ = ["./hydrostate_l40s.py"]

work_dir = "{{$HYDROSTATE_WORK_DIR:/fossfs/xiaozhen/HydroState/work_dirs/hydrostate_staged}}"
load_from = None
resume = False

model = dict(
    type="StagedHydroStateModel",
    decoder=dict(
        type="StagedTaskDecoder",
        merge_ops=True,
        fusion="add",
        attention=dict(kind="cbam", reduction=16),
        adapter=dict(kind="bottleneck", reduction=4),
    ),
    interaction=dict(type="TaskGatedInteraction", mode="sigmoid", residual_scale=0.1,
                     merge_ops=True),
    auxiliary_weight=dict(water=0.2, soil_moisture=0.2, et=0.2),
)
