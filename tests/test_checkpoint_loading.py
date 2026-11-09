"""Training checkpoint metadata must survive the current PyTorch loading defaults."""

import logging

import numpy as np
import pytest
import torch
from mmengine.logging import HistoryBuffer
from mmengine.runner import Runner
from mmengine.runner.checkpoint import CheckpointLoader

from hydrostate.engine.runner import HydroStateRunner


def test_local_checkpoint_restores_weights_history_and_optimizer(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    history = HistoryBuffer([3.0, 2.0], [1, 1])
    path = tmp_path / "epoch_1.pth"
    torch.save(dict(state_dict=expected, optimizer=optimizer.state_dict(),
                    message_hub=dict(log_scalars={"train/loss": history}),
                    meta=dict(epoch=1, iter=2)), path)
    # Use the actual Runner load path, without constructing an unrelated training setup.
    runner = object.__new__(HydroStateRunner)
    runner.model = torch.nn.Linear(2, 1)
    runner._hooks = []
    runner.logger = logging.getLogger("checkpoint-test")
    default_loader = CheckpointLoader._schemes[""]
    restored = runner.load_checkpoint(path, strict=True)
    assert runner._has_loaded
    for name, value in runner.model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    np.testing.assert_array_equal(
        restored["message_hub"]["log_scalars"]["train/loss"].data[0], [3, 2]
    )
    resumed_optimizer = torch.optim.AdamW(runner.model.parameters())
    resumed_optimizer.load_state_dict(restored["optimizer"])
    assert resumed_optimizer.param_groups[0]["lr"] == 0.01
    assert restored["meta"] == dict(epoch=1, iter=2)
    assert CheckpointLoader._schemes[""] is default_loader


def test_remote_checkpoint_policy_is_unchanged(monkeypatch):
    received = []
    monkeypatch.setattr(Runner, "load_checkpoint", lambda self, path, **kw: received.append(path))
    runner = object.__new__(HydroStateRunner)
    runner.load_checkpoint("https://example.org/model.pth")
    assert received == ["https://example.org/model.pth"]


def test_missing_checkpoint_is_not_silently_ignored(tmp_path):
    runner = object.__new__(HydroStateRunner)
    with pytest.raises(FileNotFoundError):
        runner.load_checkpoint(tmp_path / "missing.pth")
