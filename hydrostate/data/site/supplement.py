"""Freeze an additive plan using existing stations and cached static ET evidence."""

import json
from pathlib import Path

import pandas as pd

from hydrostate.data.gee.contracts import atomic_json, fingerprint

from .archives import sha256
from .spatial import static_quality


def prepare_supplement(plan, baseline, output, config, frequency_bin=1):
    plan, baseline, output = map(Path, (plan, baseline, output))
    if output.exists():
        raise FileExistsError(output)
    selected = pd.read_parquet(plan / "selected.parquet")
    candidates = pd.read_parquet(plan / "input_candidates.parquet")
    members = pd.read_parquet(plan / "label_members.parquet")
    successes = pd.read_parquet(baseline / "samples.parquet")
    scope = candidates[candidates.frequency_bin.eq(frequency_bin)]
    original = selected[selected.frequency_bin.eq(frequency_bin)]
    if not set(scope.physical_group_id).issubset(set(original.physical_group_id)):
        raise ValueError("Supplement would introduce new stations")
    recovered = []
    for row in original.to_dict("records"):
        path = baseline / "rejected" / f"{row['candidate_id']}.json"
        if not path.exists():
            continue
        rejection = json.loads(path.read_text())
        if rejection["reason"] != "no_accepted_spatial_operator":
            continue
        for quality in rejection.get("spatial", {}).values():
            identity = quality.get("query_identity")
            if identity is None:
                continue
            raw = json.loads(
                (baseline / "cache/static" / f"{fingerprint(identity)}.json").read_text()
            )
            revised = static_quality(
                raw["label"], raw["confidence"], raw["observed"],
                quality["source_landcover"], config,
            )
            if revised["accepted"]:
                recovered.append(row["candidate_id"])
                break
    unused = scope[~scope.candidate_id.isin(selected.candidate_id)]
    # Every eligible date is attempted: shuffling changes order, not final representation.
    unused = unused.sample(frac=1, random_state=config["seed"])
    ids = recovered + unused.candidate_id.tolist()
    if len(ids) != len(set(ids)) or set(ids) & set(successes.candidate_id):
        raise ValueError("Duplicate baseline/supplement candidate")
    chosen = candidates.set_index("candidate_id").loc[ids].reset_index()
    combined = pd.concat([successes, chosen], ignore_index=True)
    if combined.groupby("physical_group_id").split.nunique().gt(1).any():
        raise ValueError("Supplement changes fixed station split")
    chosen["draw_order"] = range(len(chosen))
    output.mkdir(parents=True)
    chosen.to_parquet(output / "selected.parquet", index=False)
    chosen.to_parquet(output / "input_candidates.parquet", index=False)
    members[members.input_candidate_id.isin(ids)].to_parquet(
        output / "label_members.parquet", index=False
    )
    report = dict(
        frequency_bin=frequency_bin, recovered=len(recovered), unused=len(unused),
        total=len(chosen), recovered_ids=recovered, minimum_date_gap_days=0,
        baseline=str(baseline), baseline_manifest_sha256=sha256(baseline / "samples.parquet"),
        parent_plan_sha256={p.name: sha256(p) for p in plan.glob("*.parquet")},
        by_split=chosen.split.value_counts().to_dict(),
        class_confidence=config["class_confidence"],
        homogeneous_fraction=config["homogeneous_fraction"],
        spatial_valid_fraction=config["spatial_valid_fraction"],
        automatic_publication=False,
    )
    atomic_json(output / "supplement.json", report)
    return report
