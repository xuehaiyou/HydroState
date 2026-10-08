"""Prepare small, CPU-only temporal metadata before copying imagery to CUDA."""

from dataclasses import dataclass
from datetime import UTC, datetime

import torch


@dataclass(frozen=True)
class ModalityMetadata:
    start: int
    stop: int
    timestamps: tuple[tuple[datetime, datetime], ...]


def prepare_temporal_metadata(sensor_valid, observation_time):
    """Preserve input order and real timestamps; rslearn still aligns modalities.

    Direct callers may supply CUDA tensors: copy each whole metadata tensor once,
    never inspect individual CUDA scalars or dynamically select CUDA indices.
    """
    if sensor_valid.ndim != 3 or sensor_valid.shape[-1] != 3:
        raise ValueError("sensor_valid must have shape B,T,3")
    if observation_time.shape != sensor_valid.shape:
        raise ValueError("observation_time must match sensor_valid shape B,T,3")
    validity = sensor_valid.detach().to(device="cpu", dtype=torch.bool).tolist()
    times = observation_time.detach().to(device="cpu", dtype=torch.long).tolist()
    indices, samples = [], []
    for batch_index, (sample_valid, sample_times) in enumerate(zip(validity, times)):
        modalities = []
        for modality in range(3):
            start = len(indices)
            timestamps = []
            for index, valid in enumerate(sample_valid):
                if not valid[modality]:
                    continue
                timestamp = datetime.fromtimestamp(
                    sample_times[index][modality], tz=UTC
                ).replace(tzinfo=None)
                indices.append(index)
                timestamps.append((timestamp, timestamp))
            modalities.append(ModalityMetadata(start, len(indices), tuple(timestamps)))
        if all(item.start == item.stop for item in modalities):
            raise ValueError(f"sample {batch_index} has no valid satellite modality")
        samples.append(tuple(modalities))
    return tuple(samples), torch.tensor(indices, dtype=torch.long)
