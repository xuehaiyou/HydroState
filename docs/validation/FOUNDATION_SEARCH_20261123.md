# 三种基础模型架构的站点与人工 Cover 配对实验

固定参照为 [B 的六轮实验](MANUAL_COVER_TOP3_RESULTS_20261123.md)。从 HPC2021 已执行的
`manual_cover_top3_20261122/training/B/epochs_1_6/config.py` 派生运行配置。
复用训练/验证清单 10,912 / 3,681 个窗口、预计算统计量、七个输出、空间支持损失、
人工 Cover 监督与联合选模，种子 3407，有效 batch 10。开发验证比较同一所选 epoch
的站点 SM/ET RMSE、Pearson R，冰川三类分开的比例误差、F1 和支持数，以及洪水
水体比例误差、IoU 和召回率。留出的完整测试集待模型定型后使用。

| 实验 | HydroState 中实际执行的变化 |
|---|---|
| DiTASK | 在 OlmoEarth 各层 Q/K/V/O 投影上对冻结的奇异谱做单调位置变换。连续分段仿射速度场用缩放平方近似积分，零速度对应恒等变换。第 4/8/12 层输出 Cover、SM、ET 专属分支。复用 B 的时间融合权重结构与任务读出。实现使用原生 PyTorch 和一次性 SVD，属于论文机制的迁移。 |
| D2MoE | 冻结 OlmoEarth 主干，全部注意力层加入 rank 16 共享更新。三个深度出口分别以有效 token 的均值进行任务条件路由，每个样本从四个 rank 4 专家中选两个，对 Q/K/V/O 加权。未适配的冻结主干额外前向提供最深层先验，各任务通过通道门控注入三个深度，再使用 B 的读出。三个同网格深度和先验融合均为 OlmoEarth 适配。 |
| AnySat | 使用官方 AnySat Base 权重及代码 `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`。每个 8 天槽保留实际可用传感器，使用遮罩与年内日序。按官方顺序初始化光学波段映射，S2 的两个额外波段可通过可训练线性输入映射参与；S1 第三通道初始化为标准化雷达对比。使用全部 11 个 Landsat L1 TOA 通道。输入归一化沿用 HydroState 的已存标度并重新居中，不主张与上游训练分布相同。每槽提取两层中间特征及最终融合特征，槽间聚合由 B 的时间解码器承担。训练输入映射和末端跨模态块，其余 AnySat 权重冻结。 |

DiTASK 和 D2MoE 对各任务出口分别调用同一套时间融合参数。AnySat 改变了主干、
预训练权重、槽内模态融合方式及可训练参数范围，结果代表整套方案。
同一传感器、时间和标签清单可用于三个候选，未另设仅 S1/S2 的基线。

参考 [DiTASK 官方实现](https://github.com/ipsitmantri/DiTASK)、
[AnySat 官方实现](https://github.com/gastruc/AnySat)、
[D2MoE 论文](https://arxiv.org/abs/2611.02769)。

## HPC2021 提交与启动验收

根目录为 `/scr/u/xiaoz/HydroState/experiments`。三个实验独立排队。每个实验按
实测速度分两段提交，第二段通过 Slurm afterok 和 epoch checkpoint 接续。每项均在单张
L40S 上使用同一真实样本预检通过 4 次训练更新和 3 个验证批次，有效训练 batch=10
（5 × 2），验证 batch=8。

| 方案 | 冻结训练代码 | 预检 | 正式作业与轮次 | 相对于根目录的输出 |
|---|---|---|---|---|
| DiTASK | `c5eafa7` | 4494695 | 4494703（1–4）、4494704（5–6） | `foundation_20261123/ditask` |
| AnySat | `c5eafa7` | 4494696 | 4494718（1–3）、4494719（4–6） | `foundation_20261123/anysat` |
| D2MoE | `84ed241` | 4494717 | 4494720（1–3）、4494722（4–6） | `foundation_20261123_retry/d2moe` |

D2MoE 首次预检 4494697 暴露 BF16 路由张量与 FP32 softmax 的 scatter 类型不一致；
修复后重新通过。各目录的 `bench_5.json` 保存预检结果，`epochs_*/submission.json`
保存提交参数。基线继续使用已有运行。前两项使用修复前已冻结的代码。
