# MLoRE block

Author: Yuqi Yang. License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Source: [official MLoRE implementation](https://github.com/YuqiYang213/MLoRE/blob/5817007c6b73cc4393d242d3a99c7a7e0b77e6ab/models/transformers/MLoRE.py).
Pinned commit: `5817007c6b73cc4393d242d3a99c7a7e0b77e6ab`.

`vendor.py` contains unchanged class bodies for `LoraBlock`, `SpatialAtt`, and `MOEBlock`.
Only imports and the BatchNorm alias needed by those classes are retained; the upstream
ViT, timm dependencies, training pipeline and output heads are not included.
The module retains noisy train-time top-k routing, shared low-rank convolutional experts,
private low-rank convolution, shared convolution with detached input, and per-task BatchNorm.

HydroState's wrapper applies input/output validity masks and uses the three private
Coverage/SM/ET features at the recipient interval. One block is added after the existing
three-depth fusion. The original uses two blocks at four feature depths and 15 experts;
this screening adaptation uses four ranks (4, 8, 12, 16), top-2, private rank 8 and
64-channel inputs on a 30 × 30 grid. It is **not a full MLoRE reproduction**.
The fixed spatial gate is checked explicitly; no silent resizing is performed.

BatchNorm sees zero-filled invalid input regions; it is not a masked normalization layer.
This retains upstream behavior and is a limitation for heavily missing observations.
Top-k weights do not imply sparse execution: upstream computes every expert output.
No speedup claim or universal negative-transfer guarantee follows from this candidate.
Weights and inference accuracy require separate real-data validation.

## Complete decoder (2026-11-18)

`decoder.py` implements the complete active decoder path of the same pinned official
`MLoRE_vitB.py`: four selected depths, two MOEBlocks per depth, task-specific learned
four-depth masks, 4x bilinear feature upsampling, then official ConvHead. ConvHead's
class body is included unchanged in `vendor.py`. The official forward/cal_task_feature
bodies are retained as a test-only numerical reference in
`tests/fixtures/mlore_official_reference.py`, with the same CC BY-NC 4.0 attribution.
All-valid forward output agrees within FP32 roundoff; expert classes remain unchanged.
Unused upstream q_fuse, classify_route and auxiliary-head objects are omitted because
there is no active forward/loss consumer in this selected official configuration.

The PASCAL ViT-B configuration is retained: 640 channels, 15 shared ranks 16..128
(step 8), private rank 64, top-9, pre_softmax=False. The sum of eight squared expert
importance CVs has coefficient 0.0001, matching official losses/loss_schemes.py.
It is an additional architecture-specific regularizer, not a station loss replacement.

Explicit task-interface differences:
- OlmoEarth shared-LoRA replaces the RGB ViT. Exits are blocks 3/6/9/12. Existing
  encoder normalization at every exit is retained; temporal attention queries each
  depth separately before the decoder's final-depth LayerNorm. This is not the
  original RGB benchmark protocol or a claim of reproducing its published scores.
- Three task groups have 3/1/3 output channels; output logits are resized from the
  120x120 head grid to the 40x40 retrieval grid, then existing physical activations
  and support-aware product/station losses apply.
- Input validity is applied at the temporal interface and final depth fusion.
  Entirely unobserved windows are excluded from balancing. BatchNorm inside official
  blocks is unchanged and sees zero-filled unavailable areas.
- Current experiment requires the seven output tasks to share a target interval.
- Training initialization, optimizer, supervision and screening budget use the matched
  HydroState experiment, not the original PASCAL optimizer/data schedule.

The existing one-block candidate remains unchanged and separately named.
