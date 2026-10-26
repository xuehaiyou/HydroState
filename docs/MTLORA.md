# 当前 MTLoRA / shared-LoRA 模型及全量微调对照

当前数据与监督以 [流程规范](RETRIEVAL_PIPELINE.md) 为准。
配置选择：`configs/hydrostate_mtlora.py` 与 `configs/hydrostate_mtlora_shared.py`；
两者继承 `hydrostate_retrieval.py`，共用当前七任务、五时间槽和训练统计。
`configs/hydrostate_full.py` 提供相同任务结构的全量微调对照。

## 编码器与分支

Backbone 为 OlmoEarth-v1.2-Base：12 个 Transformer block、768 通道。
所有 block 的 Q/K/V、注意力输出投影和 MLP 线性层都有共享 TA-LoRA。
第 4、8、12 层提供任务出口（配置索引 `[3, 7, 11]`）：

- 继续传播的共享流使用原权重 + 共享低秩更新。
- 各任务出口使用原权重 + 专属低秩更新，分支位于注意力输出投影和 MLP。
- 注意力计算及 Q/K/V 共用；专属出口不继续进入下一共享 block。
- 当前共享秩为 8、专属秩为 4；缩放系数 1.0 直接乘更新，不隐含除以秩。

shared-LoRA 对照将 `private_rank=0`，保留相同的任务查询、融合、解码器和输出头。
实现采用 MTLoRA 的共享/专属思路，但 OlmoEarth 的同网格深度出口与论文的 Swin
多尺度骨干不同，不能称为论文的逐项严格复现。

## 三层融合

各任务先查询三个深度的时间特征，然后分别通过 1×1 卷积降至 64 通道；
拼接为 192 通道，再由 1×1 卷积压缩至 64 通道。
之后采用两层 3×3 卷积与 GELU 的残差块，双线性上采样，再由 3×3 卷积细化。
各任务有独立融合解码器，最终输出 30 m 网格。
这是同分辨率的不同深度特征融合，未引入 Swin 式空间金字塔。

## 参数与输入

| 模块 | 当前设置 |
|---|---|
| 原始注意力/MLP 权重和偏置 | 冻结 |
| 共享/专属 LoRA | 训练 |
| S1/S2/Landsat patch embedding | 训练；未用模态保持冻结 |
| LayerNorm affine 参数 | 训练；未用预训练 projector 保持冻结 |
| 位置/时间编码及 RoPE | 冻结 |
| 任务查询、融合、解码器、输出头及损失权重 | 训练 |
| Landsat L1 TOA 输入适配器 | 训练；不是历史 Landsat L2 adapter |
| 观测次数/时间质量融合 | 训练；末层零初始化 |

Backbone 保持 evaluation mode 禁用随机 dropout / band dropping，所选参数仍参与梯度计算。
启用非重入梯度检查点。`hydrostate_mtlora_frozen_embed.py` 仅用于冻结 patch embedding 消融。

## 全量微调配置

`hydrostate_full.py` 设置 `encoder.finetuning="full"`、两个 rank 均为 0，
不安装 LoRA 模块；第 4/8/12 层共用主干特征，各任务仍使用当前独立查询、
三层融合解码器、质量摘要、输出头和七输出监督。全部任务共用一次主干遍历。

主干保留上游原有参数可训练性，更新注意力、MLP、LayerNorm、参与输入的
patch embedding，以及原本可训练的位置/时间编码和 register 参数。
原本固定的编码、未用模态 embedding 和预训练 projector 保持冻结。
与 LoRA 对照一样，骨干保持 evaluation mode，所选参数仍正常反向传播。

优化器参考历史最新多层时序配置
`configs/experiments/legacy/hydrostate_ecosystem_multilevel_temporal_hpc.py`：

| 设置 | 全量微调 | 当前 LoRA 配置 |
|---|---|---|
| AdamW 基础学习率 | 1e-4 | 1e-4 |
| 分层学习率 | official 配方，衰减率 0.65、12 层 | 无分层衰减 |
| 权重衰减 | 0.05；损失方差参数不衰减 | 0.01 |
| 梯度裁剪 | max norm=1.0 | max norm=1.0 |
| batch / 累积 / epochs | 2 / 8 / 50 | 2 / 8 / 50 |
| 调度 | 1000 microsteps 预热，相对余弦下限 0.01 | 相同 |

因此它是沿用历史优化器配方的全量基线，不是仅切换参数冻结方式的单因素消融。
比较时固定数据清单、划分、统计量、监督定义和有效 batch，并注明优化器差异。

输入配置从契约导入 `landsat_l1_toa_native_mean_v2`。有效日期均值与 QA 摘要的定义见流程规范。

## 覆盖、SM 与 ET

WF、IF、LF 同组 softmax；SM 是非水冰陆地条件下的预测。
三个非负 ET 分支按预测覆盖比例派生总 ET，各自使用相应产品原始像元监督。

当前 `land_time_policy="require_aligned"`，不允许静默回退到其他日期的普通窗口均值。
训练 SM 使用观测水/冰覆盖，推理使用预测覆盖。样本可信陆地和已知覆盖门槛为 95%；
`min_land_fraction=0.01` 是像元级数值有效门槛，与产品支持区的 95% 质量筛选含义不同。

AdamW lr=1e-4、weight decay=0.01；默认 50 epochs，单 GPU microbatch=2、累积=8。
完整训练准备和命令见 [DATA_PREPARATION](DATA_PREPARATION.md)。
