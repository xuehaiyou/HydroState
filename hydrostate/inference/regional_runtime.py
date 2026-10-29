"""Bounded startup probes, measured batch selection, and an inference run receipt."""

import gc
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.regional.plan import load_plan
from hydrostate.datasets.regional import RegionalDataset
from hydrostate.inference.regional import load_model, predict, predict_records
from hydrostate.inference.regional_reports import check_fields

LOG = logging.getLogger(__name__)


def probe_inputs(root, dates, tasks):
    """Read eight spatially distributed windows per pilot, retaining at most 24."""
    records, report = [], []
    for date in dates:
        dataset = RegionalDataset(root, date, tasks)
        indices = np.unique(np.linspace(0, len(dataset) - 1, min(8, len(dataset))).astype(int))
        started, accepted = time.monotonic(), 0
        for index in indices:
            record = dataset[int(index)]
            for name, value in record["inputs"].items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite input {date}/{index}/{name}")
            if record["accepted"]:
                accepted += 1
                if len(records) < 24:
                    records.append(record)
        report.append(
            dict(
                date=date,
                inspected=len(indices),
                accepted=accepted,
                seconds=time.monotonic() - started,
            )
        )
        LOG.info("Input probe: %s", report[-1])
    if not records:
        raise ValueError("No accepted windows in startup probe")
    return records, report


def select_batch(results):
    usable = [r for r in results if r["state"] == "ok" and r["memory_fraction"] <= 0.85]
    if not usable:
        raise ValueError("No tested batch retains 15% GPU memory headroom")
    fastest = max(r["windows_per_second"] for r in usable)
    return min(r["batch_size"] for r in usable if r["windows_per_second"] >= fastest * 0.95)


def benchmark(model, records, device, candidates):
    if not str(device).startswith("cuda") or not candidates or min(candidates) < 1:
        raise ValueError("Batch tuning requires CUDA and positive candidates")
    results = []
    for size in sorted(set(candidates)):
        batch = [records[i % len(records)] for i in range(size)]
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        external = max(0, total - free - torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        try:
            outputs = predict_records(model, batch, device)
            for output in outputs:
                check_fields({t: output[t].numpy() for t in model.tasks})
            torch.cuda.synchronize(device)
            started = time.monotonic()
            for _ in range(3):
                predict_records(model, batch, device)
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - started
            peak = torch.cuda.max_memory_reserved(device)
            result = dict(
                batch_size=size,
                state="ok",
                windows_per_second=3 * size / seconds,
                peak_reserved_bytes=peak,
                memory_fraction=(peak + external) / total,
            )
        except torch.cuda.OutOfMemoryError:
            result = dict(batch_size=size, state="oom")
        results.append(result)
        LOG.info("Batch benchmark: %s", result)
        if result["state"] == "oom":
            break
    gc.collect()
    torch.cuda.empty_cache()
    return dict(
        selected_batch=select_batch(results),
        candidates=results,
        scope="Cached inputs, preprocessing, model and CPU transfer; excludes disk/mosaic",
        gpu=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
    )


def run(args):
    root = Path(args.root)
    run_dir = args.run_dir or root / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        state="starting",
        started_utc=datetime.now(UTC).isoformat(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    path = run_dir / "inference.json"
    atomic_json(path, receipt)
    try:
        plan = load_plan(root)
        dates = args.dates or [p["start"] for p in plan["periods"]]
        if not set(dates) <= {p["start"] for p in plan["periods"]}:
            raise ValueError("Requested dates are outside the input plan")
        for date in dates:
            manifest = json.loads((root / "manifests" / f"{date}.json").read_text())
            if manifest["plan"] != plan["identity"] or manifest["period"]["start"] != date:
                raise ValueError(f"Manifest identity mismatch: {date}")
        pilots = args.pilot_dates or []
        if not set(pilots) <= set(dates):
            raise ValueError("Pilot dates must be selected target dates")
        model, provenance = load_model(
            args.model_config, args.checkpoint, args.pretrained, args.statistics, args.device
        )
        receipt.update(plan_identity=plan["identity"], provenance=provenance)
        if args.batch_candidates:
            probe_dates = pilots or [dates[i] for i in sorted({0, len(dates) // 2, len(dates) - 1})]
            records, report = probe_inputs(root, probe_dates, model.tasks)
            measured = benchmark(model, records, args.device, args.batch_candidates)
            args.batch_size = measured["selected_batch"]
            atomic_json(run_dir / "startup.json", dict(input_probe=report, benchmark=measured))
            del records
        receipt.update(state="predicting", batch_size=args.batch_size)
        atomic_json(path, receipt)
        product = predict(
            root,
            model,
            provenance,
            dates=args.dates,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            pilot_dates=pilots,
            progress_seconds=args.progress_seconds,
            display_surface_min=args.display_surface_min,
        )
        receipt.update(
            state="complete", product=product, finished_utc=datetime.now(UTC).isoformat()
        )
        atomic_json(path, receipt)
        return dict(product=product, receipt=str(path))
    except Exception as exc:
        receipt.update(state="failed", error=str(exc), updated_utc=datetime.now(UTC).isoformat())
        atomic_json(path, receipt)
        raise
