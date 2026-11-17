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
