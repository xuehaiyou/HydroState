# 架构与实验总览

更新：2026-11-18。本文记录已有实验与待开展方案，不代表实时作业状态。
训练配置存在、GPU 验收通过、训练启动、完整精度验证完成是不同状态。
历史运行证据保留在对应验收记录；本次文档整理未查询 HPC、修改训练代码或提交任务。

2026-11-26 新增[双 FM TerraWeave 六轮架构比较](validation/TERRAWEAVE_20261126.md)，与既有人工 Cover、站点 SM/ET 联合监督协议对照。

## 评价依据：用户最新决定

- 以站点 **SM、ET** 的验证精度为主要依据，分别报告 Pearson R、RMSE 和 Bias。
- 产品对照只称为“产品一致性”，作为参考，不作为独立真值精度或架构胜负依据。
  产品仍可用于训练监督，这与是否用于精度评价是两回事。
- 后续 checkpoint 选择应只使用站点指标。建议采用训练集标准差归一化后的
  SM、ET RMSE 等权平均的负值；新搜索入口采用此聚合公式，指标名为 `site_score`。
- 既有运行使用包含产品项的 joint_score；新搜索入口已实现站点选模，历史运行保持原规则。
  需要重新检查各已保存 checkpoint，不能只更改表头便宣称完成重新选模。
- Pearson R 使用预测与观测的配对值或充分统计量计算，不由 R² 开平方得到。
  产品与模型的观测子集、时间支持不同，不直接比较指标差值宣称提升。

## 已有实现与实验

| 架构／实验 | 结构与初始化 | 已核实证据及边界 | 入口／说明 |
|---|---|---|---|
| U-Net | 随机初始化卷积编码器—解码器，不加载 FM | 已有历史训练对照及 HPC 验收；不在本轮四模型小样本比较内 | [配置](../configs/hydrostate_unet.py)、[说明](UNET.md)、[HPC 记录](validation/UNET_HPC_20261102.md) |
| MTLoRA | 共享／专属 LoRA，三个编码深度的任务出口 | 已有实现与历史验证；是 OlmoEarth 适配，不是原论文 Swin 多尺度架构的严格复现 | [配置](../configs/hydrostate_mtlora.py)、[说明](MTLORA.md) |
| Shared-LoRA A | 共享 LoRA，独立任务查询与解码；FM 预训练初始化 | 已有产品-only 历史实验及产品＋站点联合实验；本轮小样本六轮验证完成，二者不能混为同一监督设置 | [联合配置](../configs/hydrostate_joint_shared.py)、[说明](COSA.md) |
| CoSA independent B | 共享目标时间状态，Coverage/SM/ET 独立空间分支 | 已有联合训练；本轮小样本六轮验证完成 | [联合配置](../configs/hydrostate_joint_independent.py)、[说明](COSA.md) |
| CoSA Router C | 在 B 上增加任务间 Q–K 路由 | 已有训练与验证记录；未纳入本轮四模型小样本比较 | [联合配置](../configs/hydrostate_joint.py)、[说明](COSA.md) |
| AdapterFusion D | 加载并冻结 A 的已训练专家，仅训练融合模块与残差系数 | GPU 验收通过，已提交正式训练；额外融合阶段不能当作等预算从头对照；此处不声明最终完成 | [配置](../configs/hydrostate_joint_adapterfusion.py)、[验收](validation/ADAPTERFUSION_HPC_20261115.md) |
| AdapterFusion E | FM 保留预训练，任务模块与融合模块新初始化后联合训练 | 已有训练启动记录，作为独立实验；不在本轮四模型比较内，此处不声明最终完成 | [配置](../configs/hydrostate_joint_adapterfusion_scratch.py)、[说明](ADAPTERFUSION.md)、[搜索记录](validation/ARCHITECTURE_SEARCH_20261117.md) |
| 单层空间 CGC | 共享／任务私有专家及 softmax 门控；参考 LibMTL | 本轮小样本六轮验证完成；改用了项目空间专家及门控输入，不是完整多层 PLE | [配置](../configs/hydrostate_joint_cgc.py)、[实现边界](ARCHITECTURE_SEARCH.md#cgc-候选的实现边界) |
| 单层 MLoRE | 在 CoSA 私有特征上接一层官方专家模块，4 专家、top-2 | GPU 验收通过；最近已核实记录为第二／第四轮完整验证，不代表实时进度；不是完整论文解码器 | [配置](../configs/hydrostate_joint_mlore.py)、[来源与边界](../hydrostate/models/mlore/PROVENANCE.md) |

另有已配置但本页未确认完整精度实验的变体：
[Full 全量微调](../configs/hydrostate_full.py) 和
[MTLoRA 冻结 patch embedding 消融](../configs/hydrostate_mtlora_frozen_embed.py)。
它们是参数训练策略对照，不因配置存在就计为已完成训练。

本轮统一小样本比较是 **A、B、单层 CGC、单层 MLoRE**，固定 4,528 个训练窗口。
截至最近已核实结果，完成同等六轮验证的 A/B/CGC 中，CGC 的站点 SM、ET
Pearson R 均最高、RMSE 均最低；这是按旧 joint_score 所选 checkpoint 的比较，
仍需按新的站点选模口径复核。单种子开发集结果不是独立测试结论。
详细数值及运行出处见[小样本实验记录](validation/ARCHITECTURE_SEARCH_20261117.md)。

## 准备开展的架构

以下为本轮开展清单（2026-11-18 更新）：完整 MLoRE、PLE 及匹配 CGC 已实现并通过
本地相关检查；AdapterFusion、CoSA Router 复用既有实现。本轮 HPC 验收与训练状态
单独记录，不能以本地检查或历史运行替代本轮 GPU 验证。

| 候选 | 复现目标 | 必须明确的边界 |
|---|---|---|
| 完整 MLoRE 解码架构 | 固定官方源码版本，核对论文及补充材料；恢复多深度特征输入、多阶段解码、专家与路由、适用的辅助损失；配置由论文对应实验确定，不默认沿用缩减的 4 专家/top-2 | 用 OlmoEarth 接收多源时序数据，保留必要的目标时间接口；让 MLoRE 承担完整任务解码，避免叠加现有 CoSA 私有解码器；不是原始主干与原始数据集上的逐项复现 |
| 多层 PLE | 完整实现任务专家、共享专家、任务门控、中间层共享门控和共享分支逐层更新；层数与专家配置核对论文后确定 | 原论文面向推荐任务，遥感空间输入、专家运算与输出接口的必要改动逐项记录；不能将当前单层 CGC 直接改名为 PLE |
| 与新 PLE 配套的单层 CGC 对照 | 与 PLE 使用相同专家定义、门控输入和训练设置，关闭多层渐进提取以比较层级机制 | 这是新实现的匹配对照，不等于已有 CGC 权重；旧实验保留，不覆盖 |
| AdapterFusion | 复用已有融合实现，纳入相同小样本清单和站点 SM/ET 评价；以 E 的 FM 预训练＋新任务模块联合训练设置作为从头对照 | D 的冻结专家后追加融合属于另一训练协议，保留历史证据并单独报告累计预算，不能与 E 混为一组；本轮尚未提交 |
| CoSA Router C | 复用已有 Q–K Router，与 independent B 在相同数据、初始化和训练预算下比较任务交互的作用 | 项目自研架构，不称为外部论文复现；已有历史训练，本轮统一小样本与站点选模复验尚未提交 |

MLoRE 方法来源：[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Multi-Task_Dense_Prediction_via_Mixture_of_Low-Rank_Experts_CVPR_2024_paper.html)。
PLE／CGC 方法来源：[RecSys 2020](https://doi.org/10.1145/3383313.3412236)。
MTLoRA 方法来源：[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Agiza_MTLoRA_Low-Rank_Adaptation_Approach_for_Efficient_Multi-Task_Learning_CVPR_2024_paper.html)。
CoSA 为项目自研架构；Shared-LoRA A 为项目对照，不能直接赋予 MTLoRA 论文的发表身份。

## 后续验证顺序

1. 先核对官方实现与许可证，形成论文—代码—项目差异表，明确哪些内部结构原样保留。
2. 对未修改模块做同输入、同权重的输出核对；验证时间、空间支持、掩膜、梯度与 checkpoint。
3. 用真实数据完成短测，记录总／可训练参数、显存和吞吐，再确定正式运行预算。
4. 使用固定数据划分、训练统计、初始化来源和匹配有效 batch，统一站点选模规则。
   不复用短测任务权重作为正式初始化；不读取封存测试集作架构筛选。
5. 小样本筛选后，按统一收敛规则扩展训练预算，再扩大子集并做匹配随机种子复验。
   同时报告容量与计算差异，不把新增专家容量的全部收益归因于路由机制。

本轮完整实现和提交进度见[运行记录](validation/EXPERT_SEARCH_20261118.md)。
旧运行、checkpoint、数据和历史指标保持不变。
