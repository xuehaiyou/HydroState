"""Streaming population moments, shared by data preparation workflows."""

import math

import numpy as np


class RunningMoments:
    """Merge batch moments in float64 without retaining the training labels."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite value among valid training labels")
        n = values.size
        mean = float(values.mean())
        delta = mean - self.mean
        total = self.count + n
        self.m2 += float(np.square(values - mean).sum()) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def result(self):
        if self.count < 2:
            raise ValueError("At least two valid labels are required for standardization")
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        if std <= 1e-8:
            raise ValueError("Near-constant target; standardization needs an explicit decision")
        return dict(count=int(self.count), mean=self.mean, std=std)
