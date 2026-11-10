"""CUDA memory evidence shared by disposable probes and submission checks."""

import math


class CudaMemoryMeter:
    """Retain peaks across MMEngine logger resets; cache is diagnostic, not live demand."""

    def __init__(self):
        self.values = dict(peak_allocated_gib=0., peak_reserved_gib=0.,
                           peak_external_gib=0., gpu_total_gib=0., memory_samples=0)

    def sample(self):
        import torch

        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        current = dict(
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            # CUDA context, libraries and any other device use outside this allocator.
            peak_external_gib=max(0, total - free - torch.cuda.memory_reserved()) / 2**30,
        )
        for key, value in current.items():
            self.values[key] = max(self.values[key], value)
        self.values["gpu_total_gib"] = total / 2**30
        self.values["memory_samples"] += 1
        return dict(self.values)


def memory_requirement_gib(report):
    """Conservative sum of live-tensor peak and sampled external peak, excluding cache."""
    values = [report[k] for k in ("peak_allocated_gib", "peak_external_gib", "gpu_total_gib")]
    if (any(not math.isfinite(v) or v < 0 for v in values)
            or values[0] == 0 or values[2] == 0 or report["memory_samples"] < 1):
        raise ValueError("Missing or invalid GPU memory evidence")
    return values[0] + values[1]


def has_memory_headroom(report):
    # External use is sampled, not a guaranteed continuous maximum. Keep 10% reserve
    # and require real multi-step/OOM checks separately; reserved cache can be reused.
    return memory_requirement_gib(report) <= report["gpu_total_gib"] * .9
