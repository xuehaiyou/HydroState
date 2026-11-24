# HydroState 三组多任务微调实验

这三组实验复用四阶段 D2MoE 的数据、时间输入、七个输出、损失与选模设置。每组从 OlmoEarth v1.2 Base 预训练权重开始，一份独立作业连续训练六轮。种子 3407，训练 10,912 个窗口，验证 3,681 个窗口。有效训练 batch 为 10，验证和测试 batch 为 8。同一份 GPU 分配先用真实样本确定可用 micro-batch，再正式训练。每轮验证，按既定的 `manual_joint_score` 选择一个检查点做完整测试。

## 三个实验

| 实验 | 保留的论文机制 | HydroState 适配 |
|---|---|---|
| D2MoE v2 + NexusFlow | 保留四阶段 D2MoE 的共享 LoRA、每阶段 16 选 8 专家、冻结先验和原预测头。在三个任务组的头前各接一个训练用代理网络。采用论文三任务附录的三层 MLP 聚合器、四层仿射耦合，以及式 4 的中心对齐。 | 每个代理读取对应任务在同一目标窗的二维状态图，按有效输入在 2×2 区域池化成固定长度，再投影到 256 维。三组的欧氏距离按任务求和、按有效样本取均值。实验固定损失系数 0.1，从第一轮启用。预测只走 D2MoE 原路径。 |
| FAAR | 按论文式 2 使用带可学习幅度的 DoRA。共享和专属适配器从 rank 64 开始随机采前缀。每个优化步用一阶参数与梯度贡献更新 EMA，每轮以 95% 的累计重要度缩秩。完整解码器含分通道径向频带滤波、跨任务低频和边缘共识及四层 HRNet 读出。 | 在第 3、6、9、12 层的输出投影及 FFN 产生任务出口。以目标时间聚合后的同网格特征替代 Swin 的空间金字塔。按实际 FFT 网格，从 DC 起划分 16 个径向频带。保留作者实现的 1 个余量槽及最小 rank 2，检查点保存活跃槽、上限与 EMA。 |
| RUTaL | 每个 FFN 共享上投影残差，下投影按 token 对四个 rank-8 残差专家做 softmax 加权；冻结原主干。初始 patch 表示进入任务阶梯，每阶段有各任务的 TAS 投影与调制器、跨任务共享的 LAE。LAE 用六头、四采样点的可变形交叉注意力及调制 FFN。使用四级 SegFormer 式任务解码器。 | 按论文的普通 ViT 设置，将 12 层等分为四段。初始 patch 图和四级编码图先经过带时间、质量掩膜的目标窗聚合，再在二维网格上运行任务阶梯。正式宽度 768，阶梯宽度 192。预测头对齐 40×40 输出网格。 |

NexusFlow 三任务主文和附录报告的耦合层数不同。此实验取附录 B 中表现最好的四层、使用其三任务中心损失。FAAR 论文给出显式 DoRA 归一化，作者公开的 `lora7.py` 前向采用加性更新，此处以论文公式为准。RUTaL 依据论文公式 5–7、§3.4 和 §4.1 实现。三个实验均沿用 HydroState 的六轮预算和优化设置，比较结果只适用于本轮训练协议。[NexusFlow 论文与附录](https://arxiv.org/html/2512.06251v1) · [作者仓库](https://github.com/ark1234/NexusFlow/tree/c176153090967f1fea42636c2bca32e2ea0dedcd) · [FAAR 论文](https://arxiv.org/html/2603.20403v1) · [FAAR 作者仓库](https://github.com/Klodivio355/FAAR/tree/ff5030f07e29d6144a87e941571e227ff38f0df1) · [RUTaL 原文](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7322.pdf)

## 比较和追溯

NexusFlow 的直接对照为已有的原版四阶段 D2MoE。FAAR 与 RUTaL 按完整方法比较。对每个验证选中的检查点分别报告冰川水体、冰雪、陆地三类指标，洪水比例误差及水体分类指标，站点 SM、ET 的 RMSE 和 Pearson R。产品一致性单列。六轮单种子比较可用于筛选候选，机制归因需进一步消融或重复实验。

复现入口是 `python -m hydrostate.experiments.dense_adaptation ARM REFERENCE RUN_DIR`，REFERENCE 为平台上已核验的 D2MoE v2 六轮配置。运行目录中的 `submission.json` 保存作业号和代码提交，`train_config.py` 保存实际 batch 与训练设置，`work/` 保存训练和验证结果，`final_test/` 保存测试结果。
