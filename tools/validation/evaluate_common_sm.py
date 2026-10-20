"""Reevaluate fixed checkpoints on identical observed-land SM support."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

import hydrostate
from hydrostate.cli import configure_target_statistics
from hydrostate.datasets.zarr_dataset import hydro_collate
from hydrostate.models.hydrostate_model import supervision_values
from hydrostate.models.water_soil_coupling import WaterSoilCoupling
from hydrostate.registry import DATASETS, MODELS

RUNS = {
    "merged_n20000_4241518": 17,
    "staged_light_land_sm_b32_n20000_4245730": 32,
    "multilevel_n20000_4251319": 32,
}


def sm_collate(batch):
    # Preserve original batch composition: legacy temporal pooling can depend
    # on the padded sequence length of other samples in the same batch.
    return hydro_collate(batch) if any(
        x["data_samples"]["soil_moisture_valid"].any() for x in batch) else None


def metrics(rows, prediction):
    y = rows.target.to_numpy(dtype=np.float64)
    p = rows[prediction].to_numpy(dtype=np.float64)
    e = p - y
    assert len(y) > 1 and np.isfinite(e).all()
    return dict(n=len(y), r2=float(1 - (e**2).sum() / ((y-y.mean())**2).sum()),
                rmse=float(np.sqrt((e**2).mean())), mae=float(abs(e).mean()),
                bias=float(e.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1, "Expose exactly one GPU"
    torch.set_float32_matmul_precision("highest")
    init_default_scope("hydrostate")
    coupling = WaterSoilCoupling(enabled=True, supervision_source="observed",
                                min_land_fraction=.01, min_window_land_fraction=.01)
    tables = {}
    for run, epoch in RUNS.items():
        dest = args.output / f"{run}.csv"
        if dest.exists():
            tables[run] = pd.read_csv(dest)
            continue
        cfg = Config.fromfile(str(args.configs / f"{run}.py"))
        configure_target_statistics(cfg, training=False)
        cfg.model.encoder.model_path = str(args.storage / "pretrained/OlmoEarth-v1_2-Base")
        cfg.val_dataloader.dataset.data_root = str(args.storage / "samples_merged_v1")
        cfg.val_dataloader.dataset.manifest = str(args.storage / "samples_merged_v1/samples.parquet")
        dataset = DATASETS.build(cfg.val_dataloader.dataset)
        model = MODELS.build(cfg.model).cuda().eval()
        checkpoint = args.storage / "checkpoints/sm_common" / run / f"best_hydro_score_epoch_{epoch}.pth"
        load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
        loader = DataLoader(dataset, batch_size=cfg.val_dataloader.batch_size,
                            shuffle=False, num_workers=2,
                            collate_fn=sm_collate)
        records = []
        with torch.inference_mode():
            for batch_idx, raw in enumerate(loader):
                if raw is None:
                    continue
                data = model.data_preprocessor(raw, training=False)
                outputs = model._forward(data["inputs"])
                samples = data["data_samples"]
                support = coupling.prepare(outputs, samples, supervision=True)
                common, target, valid = supervision_values(
                    "soil_moisture", outputs["soil_moisture"], samples, support)
                original_coupling = getattr(model, "water_soil_coupling", None)
                original_support = support if original_coupling is not None and original_coupling.enabled else None
                original, _, original_valid = supervision_values(
                    "soil_moisture", outputs["soil_moisture"], samples, original_support)
                for i, sid in enumerate(samples["metainfo"]["sample_id"]):
                    records.append(dict(
                        sample_id=sid, target=float(target[i].item()),
                        common_prediction=float(common[i].item()),
                        original_prediction=float(original[i].item()),
                        common_valid=bool(valid[i].item()),
                        original_valid=bool(original_valid[i].item()),
                        support_sha256=hashlib.sha256(
                            support.weights[i].cpu().numpy().tobytes()).hexdigest()))
                if batch_idx % 20 == 0:
                    print(run, batch_idx, "/", len(loader), "SM samples:", len(records), flush=True)
        frame = pd.DataFrame(records)
        assert not frame.sample_id.duplicated().any()
        frame.to_csv(dest, index=False)
        tables[run] = frame
        del model, loader, dataset, outputs, data, raw
        gc.collect()
        torch.cuda.empty_cache()
        print("FINISHED", run, flush=True)
    common_ids = set.intersection(*[
        set(t.loc[t.common_valid, "sample_id"]) for t in tables.values()])
    aligned = {run: t.set_index("sample_id").loc[sorted(common_ids)] for run, t in tables.items()}
    baseline = next(iter(aligned.values()))
    for t in aligned.values():
        assert np.array_equal(t.target.values, baseline.target.values)
        assert (t.support_sha256.values == baseline.support_sha256.values).all()
    result = dict(
        protocol="Observed water fraction; land weights=1-WF; pixel land>0.01; window land>=0.01; complete observed support; intersection of valid sample IDs.",
        checkpoint_selection="Original validation hydro_score best epoch, fixed before reevaluation.",
        inference="Float32, highest matmul precision; original validation batch size and composition.",
        common_sample_count=len(common_ids),
        common_ids_sha256=hashlib.sha256("\n".join(sorted(common_ids)).encode()).hexdigest(),
        models={run: dict(epoch=RUNS[run],
                          common=metrics(aligned[run], "common_prediction"),
                          original=metrics(t.loc[t.original_valid], "original_prediction"))
                for run, t in tables.items()})
    (args.output / "common_sm_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
