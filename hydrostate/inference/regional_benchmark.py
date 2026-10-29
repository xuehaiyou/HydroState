"""Read-only input/model profiling; scratch accumulators never publish region products."""

import gc
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, default_collate

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.data.retrieval.alignment import align_inputs
from hydrostate.datasets.regional import RegionalDataset
from hydrostate.datasets.retrieval_dataset import as_tensors
from hydrostate.inference.regional import RegionalMosaic, load_model, predict_records
from hydrostate.inference.regional_reports import check_fields

LOG = logging.getLogger(__name__)


def read_probe(dataset, start, count):
    """First-touch versus repeated reads; do not claim that the OS cache is cold."""
    records, measurements = [], []
    for repeat in range(2):
        native_seconds = align_seconds = 0.0
        for index in range(start, start + count):
            clock = time.perf_counter()
            arrays, accepted = dataset.native_inputs(index)
            native_seconds += time.perf_counter() - clock
            clock = time.perf_counter()
            inputs = as_tensors(align_inputs(arrays))
            inputs["target_time"] = {
                task: torch.from_numpy(arrays["target_bounds"].copy()) for task in dataset.tasks
            }
            align_seconds += time.perf_counter() - clock
            if not repeat and accepted:
                records.append(dict(inputs=inputs, data_samples={}, index=index, accepted=True))
        measurements.append(
            dict(
                repeat=repeat,
                windows=count,
                native_seconds=native_seconds,
                alignment_seconds=align_seconds,
                windows_per_second=count / (native_seconds + align_seconds),
            )
        )
        LOG.info("Reader %s: %s", dataset.period["start"], measurements[-1])
    return records, measurements


def timed_batch(model, records, device):
    """Synchronize phase boundaries for attribution, outside the throughput measurement."""
    torch.cuda.synchronize(device)
    clock = time.perf_counter()
    batch = default_collate(records)
    data = model.data_preprocessor(dict(inputs=batch["inputs"], data_samples={}), training=False)
    torch.cuda.synchronize(device)
    prepared = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.predict(data["inputs"])
    torch.cuda.synchronize(device)
    predicted = time.perf_counter()
    values = torch.stack([torch.stack([o[t] for t in model.tasks]) for o in outputs]).float().cpu()
    masks = torch.stack(
        [
            torch.stack([o[k] for k in ("prediction_valid", "soil_moisture_prediction_valid")])
            for o in outputs
        ]
    ).cpu()
    torch.cuda.synchronize(device)
    transferred = time.perf_counter()
    for output in values:
        check_fields({t: value.numpy() for t, value in zip(model.tasks, output)})
    del masks, values, outputs, data
    return dict(
        preprocess_seconds=prepared - clock,
        predict_seconds=predicted - prepared,
        transfer_seconds=transferred - predicted,
    )


def batch_probe(model, records, device, sizes, repeats):
    results = []
    for size in sizes:
        batches = [records[i : i + size] for i in range(0, len(records), size)]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            stages = timed_batch(model, batches[0], device)
            torch.cuda.synchronize(device)
            clock = time.perf_counter()
            for _ in range(repeats):
                for batch in batches:
                    predict_records(model, batch, device)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - clock
            result = dict(
                batch=size,
                state="ok",
                windows_per_second=len(records) * repeats / seconds,
                evaluated_windows=len(records) * repeats,
                maximum_batch=min(size, len(records)),
                seconds=seconds,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                **stages,
            )
        except torch.cuda.OutOfMemoryError:
            result = dict(batch=size, state="oom")
        results.append(result)
        LOG.info("Cached batch: %s", result)
        if result["state"] == "oom":
            break
    gc.collect()
    torch.cuda.empty_cache()
    return results


def pipeline_probe(dataset, start, count, model, device, workers, mosaic, batch_size=2,
                   spatial=False):
    """Measure exposed loader wait separately from inference and overlap accumulation."""
    indices = dataset.spatial_indices()[start:start+count] if spatial else range(start, start+count)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, num_workers=workers, collate_fn=list)
    clock = last = time.perf_counter()
    sums = dict(loader_wait_seconds=0.0, predict_seconds=0.0, mosaic_seconds=0.0)
    accepted = 0
    batches = []
    for batch_index, records in enumerate(loader):
        ready = time.perf_counter()
        usable = [r for r in records if r["accepted"]]
        outputs = predict_records(model, usable, device) if usable else []
        predicted = time.perf_counter()
        for record, output in zip(usable, outputs):
            row = dataset.rows[record["index"]]
            mosaic.add(row["row"], row["col"], output)
        finished = time.perf_counter()
        values = dict(
            loader_wait_seconds=ready - last,
            predict_seconds=predicted - ready,
            mosaic_seconds=finished - predicted,
        )
        batches.append(dict(batch=batch_index, accepted=len(usable), **values))
        for key, value in values.items():
            sums[key] += value
        accepted += len(usable)
        last = finished
    steady = batches[4:] or batches
    steady_seconds = sum(sum(b[k] for k in sums) for b in steady)
    return dict(
        workers=workers,
        spatial_order=spatial,
        batch=batch_size,
        windows=count,
        accepted=accepted,
        seconds=time.perf_counter() - clock,
        **sums,
        steady_windows_per_second=sum(b["accepted"] for b in steady) / steady_seconds,
        batches=batches,
    )


def profile_model(model, records, device, output_dir):
    handles, stacks = [], {}
    modules = {"encoder": model.encoder, "input_fusion": model.input_fusion}
    modules.update({f"query/{t}": module for t, module in model.queries.items()})
    modules.update({f"decoder/{t}": module for t, module in model.decoders.items()})
    modules.update(
        {f"block/{i}": module for i, module in enumerate(model.encoder.backbone.model.blocks)}
    )

    def before(name):
        def enter(module, inputs):
            scope = torch.profiler.record_function("section/" + name)
            scope.__enter__()
            stacks.setdefault(name, []).append(scope)

        return enter

    def after(name):
        def leave(module, inputs, output):
            stacks[name].pop().__exit__(None, None, None)

        return leave

    for name, module in modules.items():
        if module is not None:
            handles.extend(
                (
                    module.register_forward_pre_hook(before(name)),
                    module.register_forward_hook(after(name)),
                )
            )
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            predict_records(model, records[:2], device)
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(output_dir / "model_trace.json"))
        events = prof.key_averages()
        (output_dir / "operators.txt").write_text(
            events.table(sort_by="self_cuda_time_total", row_limit=35)
        )
        rows = [
            dict(
                name=e.key,
                count=e.count,
                cpu_ms=e.cpu_time_total / 1000,
                device_ms=e.device_time_total / 1000,
                self_device_ms=e.self_device_time_total / 1000,
            )
            for e in events
        ]
        return dict(
            sections=[r for r in rows if r["name"].startswith("section/")],
            top_operators=sorted(rows, key=lambda r: r["self_device_ms"], reverse=True)[:25],
        )
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    if not args.device.startswith("cuda"):
        raise ValueError("This diagnostic requires a CUDA allocation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        state="starting",
        note="Read-only inputs; no regional products are published",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    def save():
        atomic_json(output / "benchmark.json", report)

    save()
    try:
        model, provenance = load_model(
            args.model_config, args.checkpoint, args.pretrained, args.statistics, args.device
        )
        report.update(
            provenance=provenance,
            gpu=torch.cuda.get_device_name(args.device),
            gpu_total_bytes=torch.cuda.get_device_properties(args.device).total_memory,
            torch_version=torch.__version__,
            reader={},
            batches={},
        )
        datasets, retained = {}, {}
        for date in args.dates:
            dataset = datasets[date] = RegionalDataset(args.root, date, model.tasks)
            if args.start_index < 0 or args.start_index + args.windows > len(dataset):
                raise ValueError("Diagnostic window range is outside the regional index")
            records, measured = read_probe(dataset, args.start_index, args.windows)
            if not records:
                raise ValueError(f"No accepted records for {date}")
            retained[date] = records
            report["reader"][date] = measured
            report["state"] = "batch_probe"
            save()
            if args.reference_release:
                from hydrostate.inference.regional_comparison import compare_release

                report.setdefault("comparison", {})[date] = compare_release(
                    args, date, records, model, output)
                save()
            if not args.pipeline_only:
                report["batches"][date] = batch_probe(
                    model, records, args.device, args.batches, args.repeats
                )
            save()
        date = args.dates[-1]
        if args.tune:
            from hydrostate.inference.regional_runtime import benchmark

            report["tuning"] = benchmark(model, retained[date], args.device, args.batches)
            save()
        if args.pipeline_batch:
            from hydrostate.inference.regional_runtime import dense_record

            dense = dense_record(retained[date][0])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                predict_records(model, [dense]*args.pipeline_batch, args.device)
            report["dense_stability"] = dict(
                batch=args.pipeline_batch, consecutive_forwards=3,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_total_bytes=torch.cuda.get_device_properties(args.device).total_memory,
            )
        if not args.pipeline_only:
            report["profile"] = profile_model(model, retained[date], args.device, output)
        save()
        dataset = datasets[date]
        grid = dataset.plan["grid"]
        mosaic = RegionalMosaic(
            output / "scratch_mosaic",
            (grid["height"] // 3, grid["width"] // 3),
            model.tasks,
            dataset.plan["config"]["core_size"],
        )
        try:
            report["pipeline"] = []
            for workers in args.workers:
                # Fresh worker/group caches; operating-system cache is uncontrolled.
                fresh = RegionalDataset(args.root, date, model.tasks)
                count = args.pipeline_windows or args.windows
                if args.start_index + count > len(fresh):
                    raise ValueError("Pipeline probe exceeds the available windows")
                measurement = pipeline_probe(
                    fresh, args.start_index, count, model, args.device, workers, mosaic,
                    args.pipeline_batch or report.get("tuning", {}).get("selected_batch", 2),
                    args.pipeline_spatial,
                )
                report["pipeline"].append(measurement)
                LOG.info("Pipeline: %s", {k: v for k, v in measurement.items() if k != "batches"})
                save()
        finally:
            mosaic.close()
        report["state"] = "complete"
        save()
        return report
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        save()
        raise
