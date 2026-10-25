"""Input-only regional windows using the same assembly contract as training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hydrostate.data.regional.plan import blocks_for_window, epoch, load_plan, product
from hydrostate.data.retrieval.alignment import align_inputs, sensor_masks
from hydrostate.data.retrieval.contract import INPUT_GROUPS, SENSORS, TASKS, RetrievalSchema
from hydrostate.datasets.retrieval_dataset import as_tensors
from hydrostate.storage.reader import ShardCache
from hydrostate.storage.regional import product_path, ready


class RegionalDataset(Dataset):
    def __init__(self, root, target, tasks=TASKS):
        self.root, self.plan = Path(root), load_plan(root)
        path = self.root / "manifests" / (target + ".json")
        record = json.loads(path.read_text())
        if record["plan"] != self.plan["identity"]:
            raise ValueError("Manifest and regional plan disagree")
        self.period = record["period"]
        self.rows = pd.read_parquet(self.root / "targets" / (target + ".parquet")).to_dict(
            "records"
        )
        self.tasks, self.schema = tuple(tasks), RetrievalSchema()
        self.cache = None
        self.checked = set()

    def __len__(self):
        return len(self.rows)

    def native_inputs(self, index):
        if self.cache is None:
            self.cache = ShardCache(max_open=128)
        row = self.rows[index]
        orbit = json.loads(row["orbit"])
        arrays = {k: np.zeros(s.tail_shape, s.dtype) for k, s in self.schema.inputs.items()}
        arrays["slot_bounds"][:] = [[epoch(a), epoch(b)] for a, b in self.period["slots"]]
        arrays["target_bounds"][:] = [epoch(self.period["start"]), epoch(self.period["end"])]
        size = self.plan["block_size"]
        for name, spec in INPUT_GROUPS.items():
            scale = spec.resolution
            for slot, bounds in enumerate(self.period["slots"]):
                for block in blocks_for_window(self.plan, row["row"], row["col"]):
                    item = product(
                        self.plan, name, bounds, block, orbit if spec.sensor == "s1" else None
                    )
                    if item["id"] not in self.checked:
                        if not ready(self.root, item):
                            raise FileNotFoundError(f"Missing regional product: {item['id']}")
                        self.checked.add(item["id"])
                    group = self.cache.get(str(product_path(self.root, item)))
                    by, bx = block[0] * size, block[1] * size
                    y0, x0 = max(row["row"], by), max(row["col"], bx)
                    y1 = min(row["row"] + 120, by + size)
                    x1 = min(row["col"] + 120, bx + size)
                    src = (
                        slice((y0 - by) * 10 // scale, (y1 - by) * 10 // scale),
                        slice((x0 - bx) * 10 // scale, (x1 - bx) * 10 // scale),
                    )
                    dst = (
                        slice((y0 - row["row"]) * 10 // scale, (y1 - row["row"]) * 10 // scale),
                        slice((x0 - row["col"]) * 10 // scale, (x1 - row["col"]) * 10 // scale),
                    )
                    for field, suffix in (
                        ("values", ""),
                        ("valid", "_valid"),
                        ("quality", "_quality"),
                    ):
                        arrays[name + suffix][(slot, slice(None), *dst)] = group[field][
                            (slice(None), *src)
                        ]
        masks = sensor_masks(arrays)
        for column, sensor in enumerate(SENSORS):
            available = masks[sensor].any(axis=(1, 2, 3))
            arrays["sensor_valid"][:, column] = available
            arrays["observation_time"][:, column] = np.where(
                available, arrays["slot_bounds"].mean(axis=1).astype("i8"), 0
            )
        cfg = self.plan["config"]
        active = sum(
            any(masks[s][t].mean() >= cfg["min_input_fraction"] for s in SENSORS) for t in range(5)
        )
        return arrays, active >= cfg["min_active_slots"]

    def __getitem__(self, index):
        arrays, accepted = self.native_inputs(index)
        inputs = as_tensors(align_inputs(arrays))
        inputs["target_time"] = {
            t: torch.from_numpy(arrays["target_bounds"].copy()) for t in self.tasks
        }
        return dict(inputs=inputs, data_samples={}, index=index, accepted=accepted)
