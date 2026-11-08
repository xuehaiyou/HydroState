import json

import numpy as np
import pandas as pd

from hydrostate.data.gee.contracts import fingerprint
from hydrostate.data.site.supplement import prepare_supplement


def test_additive_plan_recovers_confidence_failure_and_adjacent_date(tmp_path):
    plan, base, out = (tmp_path / name for name in ("plan", "base", "out"))
    plan.mkdir()
    (base / "rejected").mkdir(parents=True)
    (base / "cache/static").mkdir(parents=True)
    rows = pd.DataFrame([
        dict(candidate_id=c, physical_group_id="station", split="train", frequency_bin=1,
             target_time=t)
        for c, t in [("success", "2020-01-01"), ("retry", "2020-02-01"),
                     ("new", "2020-02-09")]
    ])
    rows.to_parquet(plan / "input_candidates.parquet")
    rows.iloc[:2].to_parquet(plan / "selected.parquet")
    rows.iloc[:1].to_parquet(base / "samples.parquet")
    pd.DataFrame({"input_candidate_id": rows.candidate_id}).to_parquet(
        plan / "label_members.parquet"
    )
    identity = {"test": "raw_evidence"}
    (base / "rejected/retry.json").write_text(json.dumps(dict(
        reason="no_accepted_spatial_operator", spatial={"et": dict(
            query_identity=identity, source_landcover="CRO")}
    )))
    (base / "cache/static" / f"{fingerprint(identity)}.json").write_text(json.dumps(dict(
        label=np.full((21, 21), 4).tolist(), confidence=np.full((21, 21), .55).tolist(),
        observed=np.ones((21, 21)).tolist()
    )))
    config = dict(seed=43, class_confidence=.5, homogeneous_fraction=.95,
                  spatial_valid_fraction=.95)
    result = prepare_supplement(plan, base, out, config)
    assert result["recovered"] == 1
    assert result["unused"] == 1
    assert pd.read_parquet(out / "selected.parquet").candidate_id.tolist() == ["retry", "new"]
    assert pd.read_parquet(base / "samples.parquet").candidate_id.tolist() == ["success"]
