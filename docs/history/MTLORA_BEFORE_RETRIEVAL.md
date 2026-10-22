> 历史记录：不作为当前执行依据。最新入口见 [当前文档](../README.md)。

# OlmoEarth MTLoRA architecture

Select `configs/hydrostate_mtlora.py` through the existing training CLI. This
adds `MTLoRAHydroStateModel`, `MTLoRAOlmoEarthEncoder`, and
`MTLoRAFusionDecoder`; existing model configurations remain available.

This is an adaptation of Agiza et al. (CVPR 2024), not a reproduction of its
Swin backbone. OlmoEarth blocks 4, 8, and 12 are logical stage endpoints with
equal spatial grids. They are different depths, not spatial scales.

## Computation

Every block adapts Q/K/V with shared TA-LoRA. Ordinary blocks also adapt the
attention output projection and both MLP linear maps with TA-LoRA. At the
three endpoints those maps expose a shared branch and independent task
branches. Attention, including QKV, masks and RoPE, executes once per block.
Only the shared output continues through the backbone. Private outputs leave
the backbone for their own task's downstream path.

Following the paper's equation 2, each linear branch is `W x + b + scale B A x`.
The continuing branch uses shared A/B; a task branch uses that task's A/B.
It is not the earlier implementation's `W + gated shared BA + private BA`.
The shared low-rank output projection/MLP at the final endpoint have no
consumer when private branches exist, so their unused parameters are frozen.

Observation tokens, acquisition times, missing observations, and padding masks
are retained at all exits. Each variable independently queries its target
instant or interval at all three depths. The three queried maps are projected
to 64 channels each, concatenated, compressed with a 1x1 convolution, processed
by a two-convolution residual module, upsampled 4x, and refined with a 3x3
convolution. Pooling to the existing output grid and task heads follow.
Each variable owns its query and fusion/decoder parameters. There is no CBAM,
cross-task interaction gate, or intermediate auxiliary prediction loss.

## Defaults and parameter policy

| Setting | Default |
|---|---|
| Backbone | OlmoEarth-v1.2-Base, 768 channels, 12 blocks |
| Logical endpoints | `out_indices=[3, 7, 11]` (zero based) |
| Shared/private ranks | 8 / 4 |
| Update scale | 1.0 (direct multiplier, not implicitly divided by rank) |
| Original attention and MLP linear weights/biases | Frozen |
| LayerNorm affine parameters | Trainable |
| S1/S2/Landsat input patch embeddings | Trainable; unused modalities stay frozen |
| Position/time encodings and RoPE frequencies | Frozen |
| Unused pretraining projector | Frozen |
| Query, fusion, decoder, output heads, loss weighting | Trainable |
| Input profile | `olmoearth_l1_v1`; no L2 adapter |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01 |
| Microbatch / accumulation | 2 / 8 |
| Epoch budget / seed | 50 / 3407 |

The backbone stays in evaluation mode to disable stochastic dropout and input
band dropping, while selected parameters still receive gradients. The model
supports non-reentrant gradient checkpointing without persistent forward hooks.
Ranks and freezing flags are exposed through `model.encoder` in the config.

## Land constraint and dates

`land_time_policy="aligned_or_window"` compares water and SM query bounds for
each sample. With observed-water supervision:

- Equal bounds: use the existing detached observed-water land weights and 1%
  pixel/window engineering cutoffs.
- Different bounds: use ordinary valid-pixel window averaging for SM; do not
  discard that SM observation or apply a different date's water label.

Prediction/export likewise uses predicted-water land support only when water
and SM output query bounds agree. Otherwise SM uses ordinary valid support.
Prediction records expose `sm_land_time_aligned`. `require_aligned` instead
raises on unequal bounds. This policy does not force tasks to share dates.
ET native-footprint and water pixel supervision are unchanged.

This mixed policy differs from historical unconditional land-weighted metrics.
All comparison runs must use the same policy and valid samples. Existing
target moments remain training-only normalization statistics; the statistics
tool computes ordinary product support, not a fitted land-conditioned transform.

## Usage

Set `HYDROSTATE_DATA_ROOT`, `HYDROSTATE_MANIFEST`, `HYDROSTATE_MODEL_PATH`,
`HYDROSTATE_TARGET_STATS`, and optionally `HYDROSTATE_WORK_DIR` to the intended
dataset/checkpoint locations. The dataset must contain the pretrained-compatible
11-channel Landsat input and actual label/query timestamps. L2 packs and older
statistics from another data contract must not be substituted silently.

Generate a fresh training-statistics file if the temporal data contract has
not already been prepared (the command refuses to overwrite an existing file):

```bash
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_mtlora.py --output "$HYDROSTATE_TARGET_STATS"
CUDA_VISIBLE_DEVICES=1 hydrostate-train configs/hydrostate_mtlora.py --launcher none
```

The GPU index is an example for the designated A100 server; expose only the
assigned GPU. This implementation does not automatically submit a training job.
Current saved full-finetuning/shared-private checkpoints have different parameter
paths and must not be loaded as strict MTLoRA training resumes. Backbone
initialization loads the original OlmoEarth weights before inserting LoRA.

Controls are available through:

- `configs/hydrostate_mtlora_shared.py`: private rank zero, same task-specific
  downstream capacity, shared stage outputs consumed by every task.
- `configs/hydrostate_mtlora_frozen_embed.py`: freeze patch embeddings.

Alternatively override individual values using `--cfg-options`, for example
`model.encoder.shared_rank=16 model.encoder.private_rank=4`.

### HPC resources and data readiness

`scripts/slurm/train_retrieval.sh mtlora` and the same script with `lora`
select the private-branch model and shared TA-LoRA control, respectively.
Both request one L40S GPU and six CPU cores, use four loader workers, and
limit numerical-library threads to one. This respects the SPG-5 guidance of
at most six CPU cores per GPU. Existing jobs are not reconfigured.

Use a committed, immutable Git release and set the required source checksum,
data, statistics, and separate work-directory environment variables shown in
the script. Create log directories before submission and pass explicit Slurm
output/error paths. Generate temporal training statistics in a CPU preparation
job first; GPU jobs may depend on its successful completion with `afterok`.
Do not submit against a collection that is still downloading: require
`_COLLECTION_SUCCESS.json`, validate the samples/packs, and verify the training
data contract before submission. The launcher repeats completion and manifest
hash checks on the allocated node. It does not wait for data or automatically
submit jobs when a collection finishes.

For an explicitly authorized deferred submission, run
`tools/watch_mtlora_collection.py <run-root>/submission.json` from the frozen
release under `nohup` on the login host. The record supplies `code_root`,
`data_root`, `environment`, and an initially empty `jobs` mapping. It polls every
five minutes without reserving GPUs. `scripts/slurm/prepare_retrieval.sh` checks
the complete 20,000-sample collection and label packs, creates training-only
moments, checks both model contracts, and reads representative train/validation/
test samples. Only successful preparation permits the two GPU submissions.
The watcher respects GPU submission quotas and records every submission intent
before calling Slurm. Ambiguous submissions or failed preparation stop it for
inspection instead of blindly retrying. Check its PID, heartbeat, receipt and
log after a host restart; this `nohup` process is not a system-managed service.

The new collection uses `centered_8d_v1`: four half-open windows relative to the
target date, `[-16,-8), [-8,0), [0,8), [8,16)` days. Each modality contributes
at most one qualified scene per slot; at least two slots must be active. Both
models consume actual observation timestamps and missing-data masks, rather
than hard-coding an 8- or 16-day spacing. Historical four-slot data are not
automatically reinterpreted as this new 32-day context.

## Verification

`tests/test_mtlora.py` uses a real small OlmoEarth encoder and tests a single
attention traversal, zero-update equivalence, isolated private gradients,
no private feedback into the shared stream, masks/padding invariance, independent
query dates, checkpoint loading, optimizer/loss steps, freezing and temporal
land-support fallback. These are CPU checks, not full-scale GPU throughput or
accuracy measurements. Regression coverage includes the existing temporal,
multilevel and shared/private architectures.
