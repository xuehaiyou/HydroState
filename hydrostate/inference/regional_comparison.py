"""Compare a frozen reference release on identical cached records and checkpoint."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from hydrostate.data.gee.contracts import atomic_json
from hydrostate.datasets.regional import RegionalDataset
from hydrostate.inference.regional import load_model, predict_records


def _outputs(model, records, device, size=2):
    return [output for i in range(0, len(records), size)
            for output in predict_records(model, records[i:i+size], device)]


def reference_worker(folder):
    """Run in a separate interpreter importing only the reference release."""
    folder = Path(folder)
    args = json.loads((folder / "arguments.json").read_text())
    records = torch.load(folder / "records.pt", weights_only=False)
    dataset = RegionalDataset(args["root"], args["date"])
    # Verify native reading/alignment against the old implementation, including
    # missing sensors and all QA/time arrays, before reusing the cached pool.
    for record in (records[0], records[-1]):
        actual = dataset[record["index"]]
        torch.testing.assert_close(actual, record, atol=0, rtol=0, equal_nan=True)
    model, provenance = load_model(
        args["model_config"], args["checkpoint"], args["pretrained"],
        args["statistics"], args["device"],
    )
    outputs = _outputs(model, records, args["device"])
    torch.save(outputs, folder / "reference.pt")
    torch.cuda.synchronize()
    clock = time.perf_counter()
    for _ in range(2):
        _outputs(model, records, args["device"])
    torch.cuda.synchronize()
    atomic_json(folder / "reference.json", dict(
        windows_per_second=2*len(records)/(time.perf_counter()-clock),
        records=len(records), batch_size=2, provenance=provenance,
        imported_source=__import__("hydrostate").__file__,
    ))


def compare_release(args, date, records, model, output):
    folder = output / ("comparison_" + date)
    folder.mkdir()
    torch.save(records, folder / "records.pt")
    options = {k: str(getattr(args, k)) for k in (
        "root", "model_config", "checkpoint", "pretrained", "statistics", "device")}
    atomic_json(folder / "arguments.json", dict(options, date=date))
    env = dict(os.environ, PYTHONPATH=str(args.reference_release))
    # Execute this stable harness by path, but import model/reader from the
    # reference Git archive. No monkeypatching the model under comparison.
    with (folder / "reference.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(folder)],
            cwd=args.reference_release, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    reference = torch.load(folder / "reference.pt", weights_only=False)
    actual = _outputs(model, records, args.device)
    torch.cuda.synchronize()
    clock = time.perf_counter()
    for _ in range(2):
        _outputs(model, records, args.device)
    torch.cuda.synchronize()
    rate = 2*len(records)/(time.perf_counter()-clock)
    difference = {}
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5, equal_nan=True)
        for name in a:
            if a[name].dtype == torch.bool:
                continue
            finite = torch.isfinite(a[name]) & torch.isfinite(b[name])
            error = float((a[name][finite]-b[name][finite]).abs().max()) if finite.any() else 0
            difference[name] = max(difference.get(name, 0), error)
        for values in (a, b):
            values["et_total"] = sum(values[t] * values[c] for t, c in (
                ("et_water", "water"), ("et_ice", "snow_ice"), ("et_land", "land")))
        torch.testing.assert_close(a["et_total"], b["et_total"], atol=2e-6, rtol=1e-5,
                                   equal_nan=True)
    baseline = json.loads((folder / "reference.json").read_text())
    report = dict(reference=baseline, optimized_windows_per_second=rate,
                  speedup=rate/baseline["windows_per_second"], max_absolute_error=difference,
                  masks_identical=True, derived_et_total_verified=True)
    atomic_json(folder / "comparison.json", report)
    return report


if __name__ == "__main__":
    reference_worker(sys.argv[1])
