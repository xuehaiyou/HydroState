"""Mix published training windows without replacement for trusted-gradient supervision."""

import hashlib
import json

import numpy as np
from mmengine.dist import get_world_size
from torch.utils.data import Sampler

from hydrostate.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class TrustedSourceSampler(Sampler):
    def __init__(self, dataset, seed=3407, batch_size=10):
        if get_world_size() != 1 or dataset.split != "train":
            raise ValueError("This sampler requires single-device training")
        self.seed, self.epoch, self.size = seed, 0, batch_size
        self.length = len(dataset)
        self.product = np.arange(len(dataset.product))
        counts = [
            json.loads(row) if isinstance(row, str) else row
            for row in dataset.product.table.valid_label_counts
        ]
        self.smap = np.array([row["soil_moisture_valid"] > 0 for row in counts])
        self.pml = np.array(
            [[row[f"et_{t}_native_valid"] for t in ("water", "ice", "land")] for row in counts]
        )
        self.sites = {}
        for name in ("sm", "et"):
            self.sites[name] = np.concatenate(
                [
                    dataset.offsets[k] + np.flatnonzero(site.table["valid_" + name].to_numpy() > 0)
                    for k, site in enumerate(dataset.sites, start=1)
                ]
            )
        self.manual = np.arange(dataset.base_length, len(dataset))
        self.ids = np.concatenate(
            [
                part.table.sample_id.astype(str).to_numpy()
                for part in (*dataset.components, dataset.manual)
            ]
        )
        if len(set(self.ids)) != self.length:
            raise ValueError("Training window IDs must be unique")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def batches(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        full, tail = divmod(self.length, self.size)
        if tail != 2:
            raise ValueError("The matched subset requires a two-window final batch")
        bins = [[] for _ in range(full + 1)]
        order = rng.permutation(full).tolist()
        sm, et = (rng.permutation(self.sites[key]).tolist() for key in ("sm", "et"))
        manual = rng.permutation(self.manual).tolist()
        products = rng.permutation(self.product).tolist()
        smap = [i for i in products if self.smap[i]]
        if len(smap) % 2 or len(smap) // 2 > len(et) or len(et) > full:
            raise ValueError("Unexpected SMAP pair or ET window counts for this subset")
        for j in range(0, len(smap), 2):
            bins[order[j // 2]].extend(smap[j : j + 2])
        unused = [i for i in products if not self.smap[i]]
        # Prefer products with two comparable native observations in a single ET output.
        pml = sorted((i for i in unused if self.pml[i].any()), key=lambda i: self.pml[i].max() > 1)
        for j, site in enumerate(et):
            batch = bins[order[j]]
            if not any(self.pml[i].any() for i in batch):
                batch.append(pml.pop())
            batch.append(site)
        used_products = {i for batch in bins for i in batch if i < len(products)}
        unused = [i for i in products if i not in used_products]
        for j, batch in enumerate(bins):
            if not batch:
                batch.append(unused.pop())
            batch.append(sm.pop())
            if j < full:
                batch.append(manual.pop())
        spare = rng.permutation(unused + sm + manual).tolist()
        for batch in bins[:-1]:
            for _ in range(self.size - len(batch)):
                batch.append(spare.pop())
            rng.shuffle(batch)
        rng.shuffle(bins[-1])
        flat = [i for batch in bins for i in batch]
        if spare or len(flat) != self.length or set(flat) != set(range(self.length)):
            raise ValueError("Epoch failed the unique/exhaustive window check")
        return bins

    def audit(self, batches):
        sm, et, manual = map(set, (self.sites["sm"], self.sites["et"], self.manual))
        counts = dict(paired_sm_batches=0, paired_et_batches=0, full_manual_batches=0)
        for batch in batches:
            prod = [i for i in batch if i < len(self.product)]
            if not prod or sm.isdisjoint(batch):
                raise ValueError("Missing a product or a station SM window")
            counts["paired_sm_batches"] += int(self.smap[prod].sum() >= 2)
            counts["paired_et_batches"] += int(not et.isdisjoint(batch) and self.pml[prod].any())
            counts["full_manual_batches"] += int(
                len(batch) == self.size and not manual.isdisjoint(batch)
            )
        if tuple(counts.values()) != (self.smap.sum() // 2, len(et), len(batches) - 1):
            raise ValueError(f"Source pairing failed: {counts}")
        ids = (self.ids[i] for batch in batches for i in batch)
        return dict(
            **counts,
            batches=len(batches),
            windows=self.length,
            ordered_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        )

    def probe_indices(self, steps):
        batches = self.batches(0)[:-1]
        et = set(self.sites["et"])

        def category(batch):
            prod = [i for i in batch if i < len(self.product)]
            if self.smap[prod].sum() >= 2:
                return "sm"
            return "ordinary" if et.isdisjoint(batch) else "et"

        # Include all three types during both warmup and measured steps.
        groups = {
            k: iter([b for b in batches if category(b) == k]) for k in ("sm", "et", "ordinary")
        }
        return [i for j in range(steps) for i in next(groups[("sm", "et", "ordinary")[j % 3]])]

    def __iter__(self):
        return iter(i for batch in self.batches(self.epoch) for i in batch)

    def __len__(self):
        return self.length
