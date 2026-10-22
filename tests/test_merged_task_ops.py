"""Merged kernels must retain the unfused model's outputs and training state."""

import copy

import pytest
import torch

from hydrostate.models.task_branches import StagedTaskDecoder, TaskGatedInteraction
from hydrostate.normalization import LEGACY_TASK_NAMES as TASKS


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def decoder(merge_ops, attention="cbam", adapter="bottleneck", fusion="add"):
    return StagedTaskDecoder(
        in_channels=8, channels=(16, 12, 8), merge_ops=merge_ops,
        attention=dict(kind=attention, reduction=4),
        adapter=dict(kind=adapter, reduction=4), fusion=fusion,
    )


def assert_parameter_gradients(reference, merged, *, rtol, atol):
    assert list(dict(reference.named_parameters())) == list(dict(merged.named_parameters()))
    for (name, left), (_, right) in zip(reference.named_parameters(), merged.named_parameters()):
        assert left.grad is not None, name
        assert right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("attention,adapter,fusion", [
    ("cbam", "bottleneck", "add"), ("cbam", "depthwise", "concat"),
    ("se", "bottleneck", "concat"), ("none", "depthwise", "add"),
    ("cbam", "none", "concat"), ("none", "none", "add"),
])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_merged_decoder_matches_outputs_input_and_all_parameter_gradients(
    attention, adapter, fusion, dtype,
):
    torch.manual_seed(913)
    reference = decoder(False, attention, adapter, fusion).to(dtype=dtype)
    merged = decoder(True, attention, adapter, fusion).to(dtype=dtype)
    merged.load_state_dict(reference.state_dict(), strict=True)
    left = torch.randn(2, 8, 3, 5, dtype=dtype, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected, actual = reference(left), merged(right)
    rtol, atol = (1e-8, 1e-10) if dtype == torch.float64 else (2e-4, 2e-7)
    probes = {task: torch.randn_like(expected[task]) for task in TASKS}
    for task in TASKS:
        torch.testing.assert_close(expected[task], actual[task], rtol=rtol, atol=atol)
    # Normalize the probe objective like a mean training loss. Unnormalized
    # sums amplify harmless float32 accumulation differences near zero gradients.
    sum((expected[t] * probes[t]).mean() for t in TASKS).backward()
    sum((actual[t] * probes[t]).mean() for t in TASKS).backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)
    assert_parameter_gradients(reference, merged, rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", ["sigmoid", "softmax"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_merged_interaction_matches_outputs_and_all_gradients(mode, dtype):
    torch.manual_seed(57)
    reference = TaskGatedInteraction(channels=8, mode=mode, merge_ops=False).to(dtype=dtype)
    merged = TaskGatedInteraction(channels=8, mode=mode, merge_ops=True).to(dtype=dtype)
    merged.load_state_dict(reference.state_dict(), strict=True)
    left = {task: torch.randn(2, 8, 3, 5, dtype=dtype, requires_grad=True) for task in TASKS}
    right = {task: value.detach().clone().requires_grad_() for task, value in left.items()}
    expected, actual = reference(left), merged(right)
    rtol, atol = (1e-9, 1e-10) if dtype == torch.float64 else (1e-5, 1e-6)
    probes = {task: torch.randn_like(expected[task]) for task in TASKS}
    for task in TASKS:
        torch.testing.assert_close(expected[task], actual[task], rtol=rtol, atol=atol)
    sum((expected[t] * probes[t]).sum() for t in TASKS).backward()
    sum((actual[t] * probes[t]).sum() for t in TASKS).backward()
    for task in TASKS:
        torch.testing.assert_close(left[task].grad, right[task].grad, rtol=rtol, atol=atol)
    assert_parameter_gradients(reference, merged, rtol=rtol, atol=atol)


def test_original_checkpoint_and_adamw_state_resume_with_merged_decoder():
    torch.manual_seed(14)
    reference = decoder(False).double()
    original_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    x = torch.randn(2, 8, 3, 5, dtype=torch.float64)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        sum(value.square().mean() for value in model(x).values()).backward()
        optimizer.step()

    step(reference, original_optimizer)
    merged = decoder(True).double()
    merged.load_state_dict(reference.state_dict(), strict=True)
    restored_optimizer = torch.optim.AdamW(merged.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(copy.deepcopy(original_optimizer.state_dict()))
    step(reference, original_optimizer)
    step(merged, restored_optimizer)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(expected, merged.state_dict()[name], rtol=1e-8, atol=1e-9)
    for left, right in zip(reference.parameters(), merged.parameters()):
        for key, value in original_optimizer.state[left].items():
            torch.testing.assert_close(value, restored_optimizer.state[right][key],
                                       rtol=1e-8, atol=1e-9)
