# HydroState 实验总览与启示

本页是项目的实验索引。按研究问题归档架构训练、同权重消融、数据与标签验证、区域推理试验。精度核对于 **2026-11-27 16:11（香港时间）**，作业进度见下表。同一实验的提交、失败重试和续训归在同一项，复用的基线只算一次。旧运行沿用当时的标签、划分和选模规则。

## 当前进度与优胜模型

**综合精度最值得保留的是单 OlmoEarth 的 BDTA + MoTE。** 按预先规定的开发联合分数，固定七项产品方差为 1 的版本目前第一（−0.4448），原版学习方差为 −0.4466。两者各自选出的检查点在完整测试上分别为 −0.5097 和 −0.5141。固定方差版测试 ET 最好；原版洪水 F1、冰川冰雪和陆地 F1 最好。这一轮没有模型在所有目标上领先，微小分差仍需多种子验证。[逐项成绩、选中轮次、有效数量与原始文件](validation/ARCHITECTURE_METRICS_20261127.csv)

现行人工 Cover＋站点协议下登记 **31 项实验**。其中 21 个联合模型完成六轮开发验证，10 个已得到完整测试。相同输入方案的恢复、备用平台提交只算一次；两个在训练前取消的 HLS 作业单列为替换记录。

| 核实进度 | 数量 | 模型 |
|---|---:|---|
| 联合训练、开发选模及完整测试完成 | 10 | 单 FM BDTA + MoTE（学习、固定产品方差）、双 FM 直接读取 BDTA（学习来源权重）、B、CoSA Router、CGC、单 FM RUTaL、D²MoE v2 + NexusFlow、FAAR、SoftCon-S TerraWeave |
| 联合六轮开发验证完成，完整测试成绩待取得 | 11 | D²MoE v1/v2、DiTASK v1/v2、单 FM AnySat、CGC 局部细节、普通背景残差、零均值局部残差、完整场景输入、场景均值、双 FM 直接读取 BDTA 等权版（续跑作业 4520001 已进入完整测试） |
| ET 单任务六轮验证完成 | 1 | CGC 仅 ET，按站点 ET RMSE 选模，开发 RMSE 1.0800 mm/day，R 0.8112 |
| 训练中 | 2 | 单 OlmoEarth TerraWeave 完成 4 轮验证；OlmoEarth + AnySat 的独立参数 TerraWeave 完成 3 轮验证 |
| 已排队 | 2 | Prithvi（S2）4520958、Prithvi + SoftCon（S2 + S1）4520959。沿用原 Zarr 和六轮协议 |
| 提前取消 | 5 | 双 FM RUTaL 两组、双 FM BDTA 预融合两组、跨级共享参数 TerraWeave 原型。后者在第 1 轮取消；四级独立参数版另列 |
| 提交后更换输入，未开始训练（不计入 31 项） | 2 | 原 HLS 光学和融合任务 4520289、4520290 已取消，由上述 S2 两项替代 |

## 目前学到了什么

1. **现有证据最支持让任务自行选择共享与私有的空间特征。** 在容量相近的对照中，把逐像元的 1×1 专家换成能看邻域的 3×3 专家，站点 SM、ET 的 Pearson R 同时从 0.5814、0.8432 升至 0.5918、0.8474，两个 RMSE 也下降。在两份冻结的 CGC 权重上，切断共享路径都使 ET 误差上升，但幅度不同。共享有用，哪条路径有用也随任务、权重和监督支持而变。[空间专家与冻结干预](validation/CGC_MECHANISM_20261121.md)
2. **局部细节与场景平均量需要分别检验。** 只让 SM 的专家权重随位置变化，在两份固定模型上都让 SM RMSE 下降约 0.0002 m³/m³，ET 预测相同。联合重训后 SM R 从 0.5918 升至 0.6081，ET R 从 0.8474 降至 0.8338。零均值局部修正在人工冰川点上降低了三类比例误差，洪水 IoU 却从 0.5913 降至 0.5737。提高一种空间支持上的精度，须到其他支持上直接验证。[门控分解](validation/CGC_FACTORIZED_GATES_20261122.md) · [背景与局部修正](validation/MANUAL_BACKGROUND_RESIDUAL_20261123.md)
3. **同一模型可同时改善多个变量，分项冠军仍然不同。** BDTA + MoTE 相对单 FM RUTaL 同时提高了独立测试的冰川三类 F1、洪水 F1、站点 SM 和 ET 的 Pearson R，并降低两项站点 RMSE。与全部候选比较，开发 SM 的 RMSE/R 由 D²MoE v1 领先，开发洪水 F1/IoU 由 DiTASK v2 领先。在当时七个完整测试模型中，FAAR 的冰川水体 F1 最高，CoSA Router 的 SM RMSE 最低；新增模型后的逐项结果见下表。这组比较同时改变了主干冻结和任务适配，单独归因仍需消融。[逐模型开发结果](validation/MULTIFM_RUTAL_20261125.md) · [BDTA + MoTE 结果](validation/BDTA_MOTE_20261125.md)

4. **方差学习改变了不同目标之间的取舍。** 在同一单 FM BDTA 配置下固定产品方差，测试 ET RMSE 从 1.2702 降到 1.2605 mm/day，SM RMSE 从 0.0936 降到 0.0921 m³/m³；洪水 F1 从 0.7357 降到 0.6685。两组均接受站点和人工 Cover 监督，改变的是七个产品输出的损失权重。当前选模公式使用覆盖比例 MAE，分类 F1 不直接参与，因而综合领先与洪水分类领先可以分属两组。[方差对照](validation/BDTA_UNIT_VARIANCE_20261126.md)
5. **增加第二个 FM 的收益随目标变化。** 相对单 OlmoEarth 学习方差版，OlmoEarth + AnySat 直接读取学习版的测试 SM R 从 0.5851 升到 0.6084，冰川水体 F1 从 0.9726 升到 0.9897，洪水 F1 则从 0.7357 降到 0.7099，ET RMSE 从 1.2702 升到 1.2893。双 FM 内部，学习／等权的开发联合分数为 −0.4472／−0.4474，开发洪水 F1 为 0.7295／0.7594。当前数据支持按变量比较来源权重，两组分差不足以确立普遍优势。[双 FM 对照](validation/DIRECT_MULTIFM_BDTA_20261126.md)
6. **雷达单路的局限已测出，作为辅助输入的贡献仍待直接对照。** SoftCon-S 的测试冰川水体召回率为 0.0345，洪水水体召回率为 0.3584，站点 SM／ET R 为 0.3143／0.4723。与多模态模型相比，传感器、主干和有效观测都发生变化。Prithvi（S2）和 Prithvi + SoftCon（S2 + S1）将使用相同光学输入；两者的共同有效样本用于比较精度，各自有效样本量用于比较覆盖。[实验与有效域](validation/TERRAWEAVE_SPECIALISTS.md)

全量正式训练使用单 OlmoEarth BDTA + MoTE、固定平方误差和全部隔离后训练窗，独立于六轮子集架构排名。[全量训练协议](validation/FULL_BDTA_FIXED.md)

## 评估口径与证据

| 比较组 | 数据与选模 | 可以回答的问题 |
|---|---|---|
| 早期多变量产品 | 旧标签契约、产品指标和当时的 hydro_score。后续 20,000 窗口产品组为训练 13,906、验证 2,718、测试 3,376。 | 在各自相同产品和支持下，比较输入、解码和预训练策略对产品重建的影响。 |
| 产品＋站点，人工 Cover 加入前 | 架构短训基准为训练 4,528、开发 2,261 个窗口，通常 6 轮。首轮按含产品项的 joint_score 选模，后续按站点归一化误差的 site_score 选模。 | 历史站点 SM、ET 的对照。相同架构按两种规则选出的权重和指标可不同。 |
| 产品＋站点＋两种人工 Cover | 固定训练 10,912、开发 3,681、完整测试 11,168 个窗口。短训 6 轮，种子 3407，有效训练 batch 10，验证和测试 batch 8。按开发联合分数选定一个检查点，用该权重报告全部指标。 | 开发集用于筛选，完整测试用于评价被选权重。单种子短训的差异仍有不确定性。 |

现行 manual_joint_score 为 Cover 比例 MAE、按训练标准差归一的 SM RMSE、按训练标准差归一的 ET RMSE 三者均值的相反数，越高越好。Cover 项给冰川和洪水等权，冰川内部给水、冰雪、陆地等权。分别记录两种 Cover 来源、站点 SM（m³/m³）、站点 ET（mm/day）。相关性取配对的 Pearson R。产品指标单列为一致性参考。[评估定义](MANUAL_COVER_TRAINING.md)

## 人工 Cover 加入后的实验

下表列出当前协议下的架构、变体和对照。除明确标注者，其余联合模型已完成六轮开发验证。逐模型的选中轮、联合分数、站点 RMSE/R、冰川分项和洪水 MAE/F1/IoU 在[同口径开发结果表](validation/MULTIFM_RUTAL_20261125.md)。原多模态对照的开发集有 1,639 条 SM、446 条 ET、2,425 个冰川有效像元和 1,274,226 个洪水有效像元。冰川开发集中水体正类为零，水体比例 MAE 衡量误报量，水体 F1 记为不可估。

| 实验与直接对照 | 结果与启示 |
|---|---|
| CGC、CoSA independent B、CoSA Router C | 首批三基线均完成独立测试。B 的开发联合分数最高，CGC 的站点 ET 误差最低。给 B 增加 Q–K Router 后，测试 SM RMSE 从 0.0909 降至 0.0906，洪水 F1 从 0.6552 降至 0.6403。[基线报告](validation/MANUAL_COVER_TOP3_RESULTS_20261123.md) |
| 普通背景残差、零均值局部残差、完整场景输入、仅让场景改变均值 | 在四种变体中，零均值版的开发联合分数最高（−0.4743）。完整场景相对零均值版降低洪水比例 MAE（0.0507→0.0479）和 ET RMSE（1.0505→1.0055），冰川三类比例误差均上升。仅传递场景均值的 ET RMSE 为 1.0233。[配对结果与空间组区间](validation/MANUAL_BACKGROUND_RESIDUAL_20261123.md) |
| CGC 局部细节版（cgc_detail）、原 CGC、仅用 ET 损失的原 CGC | 同一开发 ET 上，RMSE/R 分别为 **0.9408/0.8611、0.9982/0.8425、1.0800/0.8112**。单任务保留产品 ET 和站点总 ET 监督，用最低站点 ET RMSE 选模。原 CGC 的联合监督在本次对照有利于 ET，证据限于该训练预算和划分。HKU 运行目录为 `experiments/manual_detail_et_20261123/{cgc_detail,et_only}`。 |
| DiTASK v1 / v2、D²MoE v1 / v2、单 FM AnySat | 较贴近原文的四级版均提高了各自旧版的冰雪、陆地和洪水 F1，站点 SM、ET 的 RMSE 均上升。D²MoE v1 的开发联合分数 −0.4709、SM RMSE 0.0872、SM R 0.6183。DiTASK v2 洪水 F1 为 0.7894。AnySat 冰雪 F1 为 0.8324，SM/ET RMSE 为 0.1022/1.1580；此次同时改变了预训练、输入映射和适配路径。版本间还有专家、解码等差异。[五组实现边界](validation/FOUNDATION_SEARCH_20261123.md) |
| FAAR、单 FM RUTaL | FAAR 的开发冰川水体比例 MAE 为 0.0057，完整测试三类 F1 均超过 0.95。RUTaL 开发洪水 MAE 为 0.0425、F1 为 0.7820，完整测试 F1 为 0.5327。四窗的阶梯采样权重探查在后级观察到任务差异，尚不能据此解释测试差距。HKU 诊断为 `diagnostics/rutal_ladder_probe/4507713.log`。[方法与运行目录](validation/DENSE_ADAPTATION_20261124.md) |
| D²MoE v2 + NexusFlow 对 D²MoE v2 | 两者均完成六轮。NexusFlow 第 6 轮、原版第 5 轮入选。加入 NexusFlow 后，开发 ET RMSE 从 1.0632 降至 0.9580，洪水 MAE 从 0.1038 降至 0.0536，SM RMSE 从 0.0930 升至 0.1089。NexusFlow 在完整测试的 SM R 以 0.6127 居首，但 SM RMSE 为 0.1013。原版尚缺同口径独立测试。[训练与评估](validation/DENSE_ADAPTATION_20261124.md) |
| OlmoEarth＋AnySat 双 FM RUTaL，学习来源权重 / 固定有效来源等权 | 两组均采用两个主干各四个出口，同批样本、相同预算。两项作业已于 11 月 26 日取消。学习版保留第 1 轮权重及优化器，等权版无完整检查点。两组均无完整开发验证。[恢复与验收](validation/MULTIFM_RUTAL_20261125.md) |
| BDTA＋MoTE | 冻结 OlmoEarth，用四级双向旁路和任务专家混合做适配。六轮验证与独立测试均完成，第 6 轮入选。相对单 FM RUTaL，测试洪水 F1 提高 0.2031，SM RMSE 从 0.0965 降至 0.0936，ET RMSE 从 1.4194 降至 1.2702。开发洪水 F1 仍由 RUTaL 领先（0.7820 对 0.7555），这项收益随划分而变。[分来源验证与测试](validation/BDTA_MOTE_20261125.md) |
| 双 FM BDTA + MoTE，学习来源权重 / 固定有效来源等权 | 在单 FM BDTA 的每个深度引入第二个冻结的 FM，学习版权重随任务和位置变化。两组均按用户决定取消。[结构、协议与作业](validation/MULTIFM_BDTA_20261126.md) |
| 双 FM BDTA + MoTE，直接读取 | 双向适配分别处理两路 FM。学习和等权版本只改变 G2T 中有效来源的总注意力分配。[架构与实验](validation/DIRECT_MULTIFM_BDTA_20261126.md) |
| 专用双 FM BDTA + MoTE，学习来源权重 | SSL4EO-L ResNet-50 MoCo 读取 Landsat，TerraFM-Base 读取 S1 与 S2，在同一直接读取架构检验主干分工。沿用六轮、人工 Cover 与站点选模协议。[配置、适配与运行记录](validation/SSL4EO_TERRAFM_BDTA.md) |
| 单 FM BDTA + MoTE 产品方差固定为 1 | 第 5 轮入选。相同结构的学习方差组第 6 轮入选，逐项取舍见上文及完整测试表。[配置核验](validation/BDTA_UNIT_VARIANCE_20261126.md) |
| 单 FM BDTA + MoTE，可信观测指导产品权重 | 单卡每批 10 窗。可信监督与产品组内均方误差的基础系数为 2∶1；同输出头的梯度内积调整有效产品观测的相对权重。原短训子集、六轮开发选模和独立测试协议。此次只运行动态版，与既有基线还同时存在损失和组批差异。[协议及运行证据](validation/TRUSTED_PRODUCT_GRADIENTS.md) |
| 双 FM TerraWeave 共享参数原型、四级独立参数、单 OlmoEarth 四级对照 | 三者分开归档。阶段成绩只用于了解训练进展。双 FM 前 3 轮最佳开发分数 −0.4865；单 FM 前 4 轮中第 3 轮最佳 −0.4867。相近的早期成绩尚不能判断第二个 FM 的最终收益。[双 FM](validation/TERRAWEAVE_20261126.md) · [单 FM](validation/TERRAWEAVE_OLMO_20261127.md) |
| 专用 FM TerraWeave：SoftCon（S1）、Prithvi（S2）、双路融合 | SoftCon 在第 5 轮入选，完成完整测试。其余两组仅有真实窗口前反向验收；旧 HLS 作业未产生精度。[输入及运行记录](validation/TERRAWEAVE_SPECIALISTS.md) |

**NexusFlow 如何起作用。** 同一窗口的 Cover、SM、ET 解码特征分别进入三个训练用投影网络。有输入的位置按 2×2 分区池化，经三层 MLP 和四层可逆仿射耦合，各得到一个 256 维向量。对每个样本取三个向量的均值，向训练损失加入三者到均值的欧氏距离之和，系数为 0.1。只有部分任务有标签的窗口，也能借这条损失给三个任务分支传递梯度；物理标签仍受各自掩膜约束。推理沿用 D²MoE v2 的预测路径。可逆性适用于耦合变换，池化已经丢失了部分空间细节。[原论文](https://arxiv.org/html/2512.06251v1) · [代理网络](../hydrostate/models/cosa/nexusflow.py) · [训练接入](../hydrostate/models/cosa/foundation_v2.py)

### 十个模型的完整独立测试

九个多模态模型的测试有效数量一致（SM 7,813、ET 446），下表按联合分数排序并标粗分项最优。SoftCon 单列于末行，其有效数量为 SM 6,350、ET 404。六轮开发验证均按联合分数选定一个检查点。冰川测试中的水体正类集中于 SouthCascade，单站的高分外推到其他冰川仍有不确定性。

| 架构 | 选中轮 | 联合分数 ↑ | 冰川水 F1 ↑ | 冰雪 F1 ↑ | 陆地 F1 ↑ | 洪水 MAE ↓ | 洪水 F1 ↑ | 洪水 IoU ↑ | SM RMSE ↓ | SM R ↑ | ET RMSE ↓ | ET R ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BDTA + MoTE（固定方差） | 5 | **-0.5097** | 0.9862 | 0.9585 | 0.9351 | 0.0772 | 0.6685 | 0.5020 | 0.0921 | 0.5847 | **1.2605** | **0.7576** |
| 双 FM BDTA（学习来源权重） | 6 | -0.5122 | **0.9897** | 0.9687 | 0.9492 | 0.0787 | 0.7099 | 0.5503 | 0.0913 | 0.6084 | 1.2893 | 0.7472 |
| BDTA + MoTE | 6 | -0.5141 | 0.9726 | **0.9751** | **0.9590** | 0.0763 | **0.7357** | **0.5820** | 0.0936 | 0.5851 | 1.2702 | 0.7506 |
| B | 6 | -0.5221 | 0.9246 | 0.9263 | 0.8988 | 0.0641 | 0.6552 | 0.4872 | 0.0909 | 0.6059 | 1.3303 | 0.7202 |
| CoSA Router | 5 | -0.5225 | 0.9010 | 0.9123 | 0.8861 | 0.0638 | 0.6403 | 0.4709 | **0.0906** | 0.6019 | 1.3336 | 0.7182 |
| CGC | 6 | -0.5260 | 0.8503 | 0.8461 | 0.8256 | **0.0609** | 0.6414 | 0.4721 | 0.0922 | 0.5916 | 1.2709 | 0.7550 |
| RUTaL | 6 | -0.5546 | 0.9533 | 0.9498 | 0.9266 | 0.0945 | 0.5327 | 0.3630 | 0.0965 | 0.5656 | 1.4194 | 0.6865 |
| D²MoE v2 + NexusFlow | 6 | -0.5572 | 0.9541 | 0.9093 | 0.8760 | 0.0951 | 0.6356 | 0.4659 | 0.1013 | **0.6127** | 1.3277 | 0.7221 |
| FAAR | 5 | -0.6094 | 0.9828 | 0.9719 | 0.9539 | 0.1002 | 0.5900 | 0.4184 | 0.1064 | 0.4504 | 1.5783 | 0.7015 |
| SoftCon-S TerraWeave | 5 | -0.6871 | 0.0662 | 0.7118 | 0.7135 | 0.1112 | 0.4767 | 0.3129 | 0.1115 | 0.3143 | 1.6962 | 0.4723 |

**站点 SM 要同时看相关性与绝对误差。** 在开发集上，D²MoE v1 的 R（0.6183）和 RMSE（0.0872）同时领先，完整测试成绩待补。独立测试上，D²MoE v2 + NexusFlow 的 R 最高（0.6127），CoSA Router 的 RMSE 最低（0.0906）。双 FM 学习版（0.6084）、B（0.6059）、CoSA Router（0.6019）的测试 SM R 也超过 0.6。各指标均按站点有效观测配对计算。[完整数值与有效样本量](validation/ARCHITECTURE_METRICS_20261127.csv)

### 同一检查点的站点 ET 在开发集和测试集上的 Pearson R

在 11 月 26 日的首批 17 个完成验证的联合模型中，14 个的开发集 ET R 超过 0.8。当时已有完整测试的 7 个模型，其开发集 ET R 均超过 0.8，换到测试集后范围为 0.6865–0.7550。下表包括这 7 个模型，以及开发集 ET R 较高、尚无完整测试的 CGC 局部细节版和 D²MoE v1。R 均按相应划分的全部有效站点 ET 配对合并计算。

| 架构 | 选中轮 | 开发集 ET R | 测试集 ET R |
|---|---:|---:|---:|
| BDTA + MoTE | 6 | 0.8642 | 0.7506 |
| CGC 局部细节 | 6 | 0.8611 | 暂无完整测试 |
| D²MoE v2 + NexusFlow | 6 | 0.8538 | 0.7221 |
| D²MoE v1 | 6 | 0.8456 | 暂无完整测试 |
| CGC | 6 | 0.8425 | 0.7550 |
| RUTaL | 6 | 0.8414 | 0.6865 |
| FAAR | 5 | 0.8234 | 0.7015 |
| B | 6 | 0.8221 | 0.7202 |
| CoSA Router | 5 | 0.8103 | 0.7182 |

开发和测试各有 446 个窗口，分别来自 38 和 41 个物理站点。按固定实验清单复核，两组的站点和窗口交集均为零。同期观测 ET 的中位数分别为 1.36 和 2.38 mm/day，说明两批观测的分布不同。Pearson R 对常量偏移不敏感，单凭中位数差异无法解释 R 的下降。开发集参与逐轮选模，测试集衡量同一检查点在另一批留出站点上的表现。[划分规则与站点数](validation/SITE_ET_TARGET3000_20261110.md) · [逐检查点精度](validation/ARCHITECTURE_METRICS_20261127.csv)

## 人工 Cover 加入前的架构与机理对照

| 实验或配对 | 观察与启示 |
|---|---|
| 产品 20k 的 shared-LoRA、MTLoRA、全量微调、随机 U-Net | 各自已记录的最佳产品 hydro_score 为 −0.3732、−0.3859、−0.3796、−0.4887。该批产品重建中，较少可训练参数的 shared-LoRA 超过全量微调；U-Net 同时更换了网络和预训练条件。HKU 实验目录为 `retrieval_pair_20000_20261024`、`retrieval_full_20000_20261026`、`retrieval_unet_20000_20261102`。[微调策略](MTLORA.md) · [U-Net 验收](validation/UNET_HPC_20261102.md) |
| 全量或较长联合训练的 A、B、C、AdapterFusion D/E | A 有 25 轮站点＋产品训练；B/C 对照独立分支和任务间路由。D 从已训练 A 出发冻结专家再融合，E 从预训练 FM 和新任务层联合训练。HKU 的 `adapterfusion_joint_20261116_b32` 保留至第 8 轮权重，`adapterfusion_joint_scratch_20261117_foss` 保留至第 23 轮；冻结专家的额外训练预算、监督版本与小样本 E 分开记。[B/C 运行](validation/COSA_JOINT_20261110.md) · [D/E 定义](ADAPTERFUSION.md) · [GPU 验收](validation/ADAPTERFUSION_HPC_20261115.md) |
| 首轮 6 轮的 A、B、原 CGC、单层 MLoRE | 用旧 joint_score 选模，A 的联合分数最高（−0.5854）；CGC 的两项站点 RMSE 最低（0.0876、0.9145）。一层 MLoRE 分别为 0.0926、1.1202。[小样本清单与指标定义](validation/ARCHITECTURE_SEARCH_20261117.md) |
| 站点选模的八模型：A、B、CoSA Router、AdapterFusion E、原 CGC、匹配 CGC、两层 PLE、完整 MLoRE | 原 CGC 的 ET RMSE 0.9595；匹配 CGC 为 1.0325、PLE 为 1.2464。完整 MLoRE 降至 micro-batch 2、累积 5 次并续训完 6 轮，SM/ET RMSE 为 0.1032/1.7745。增加提取层或专家复杂度未带来一致收益，MLoRE 的 BatchNorm 还受 micro-batch 影响。HKU 原始目录为 `architecture_experts_20261118_v2/<模型>/work`，MLoRE 使用 `retry_batch2/work`。[八组清单及续训](validation/EXPERT_SEARCH_20261118.md) · [PLE 对照](PLE.md) |
| CGC＋PCGrad、CGC＋Aligned-MTL、MTAN、TADFormer、BridgeNet | 五组均完成六轮。按同一站点选模规则，五者的 SM/ET RMSE 依次为 0.0903/1.0441、0.0875/1.2346、0.0887/1.0252、0.1117/1.3016、0.1004/1.1019。Aligned-MTL 改善 SM 时牺牲 ET；这组对照尚未支持用梯度重加权普遍解决任务权衡。前三组日志在 HKU，后两组在地理系 `experiments/architecture_attention_20261119/<模型>/work`。 |
| L-TAE、U-TAE、AdaShare；CGC＋Auto-Lambda 及对应固定权重对照 | L-TAE 的站点 R 为 0.5957/0.8312。随机初始化的 U-TAE、AdaShare 短训较弱，预训练与架构因素同时变化。使用相同元学习留出组时，Auto-Lambda 的 SM/ET RMSE（0.0912/1.0758）均高于固定权重（0.0897/1.0574）。[同平台对照](validation/CGC_MECHANISM_20261121.md) |
| CGC 空间专家、固定门控、局部 SM、强制 SM 私有、冻结门控替换、私有路径续训 | 检验了共享/私有、等权、训练均值、按任务替换以及同权重整块/局部门控。历史模型把所有任务切至私有路径后 ET RMSE 从 0.9595 升至 2.0841，第二份权重同向但区间跨零。仅续训 SM 私有路径一轮，两种门控都提高 SM 相关性但增大 RMSE。总 ET 同时依赖 Cover 和各分量，切换全部路径会同时改变二者。[冻结与重训结果](validation/CGC_MECHANISM_20261121.md) |
| 共享专家间比例与私有份额分别取整块/局部的 2×2 消融；统一门控一块/两块、关闭第二块、固定局部系数 | 两份冻结权重均支持私有份额局部化对 SM RMSE 的小收益，ET 的变化方向不一致。重训两块略优于一块，但固定两块权重后关闭第二块，SM/ET RMSE 仅增 0.0000031/0.0001915，配对区间跨零。当前证据支持空间专家分工，对继续堆深的支持很弱。[因子消融](validation/CGC_FACTORIZED_GATES_20261122.md) · [堆叠与推理干预](FM_MULTIVARIABLE_ADAPTATION.md) |
| 普通背景残差与零均值局部残差的历史站点对照 | 地理系作业 39192–39197 已完成站点留出评估，两种结构后来又纳入人工 Cover 联合训练。旧站点测试中，零均值相对普通修正的站点等权 ET RMSE 下降 0.0723 mm/day，开发差值区间跨零。[历史留出及人工 Cover 试验](validation/MANUAL_BACKGROUND_RESIDUAL_20261123.md) |
| 早期共享解码、三阶段、轻量三阶段、陆地条件 SM、单深度与多深度时序、共享/私有与单任务 | 10 月 10 日同到第 6 轮，共享基线与轻量陆地 SM 版的产品 SM/ET RMSE 分别为 0.0717/0.7419 与 0.0681/0.5717。10 月 17 日旧 16 变量实验，各自选中轮的多深度版产品 SM/ET RMSE 从 0.0817/0.5709 降至 0.0758/0.5626，GPP 则退化。另有 L2 共享、共享/私有与独立 Water/SM/ET 对照。源记录为 `outputs/validation_comparison_20261010_2044.json`、`outputs/training_accuracy_20261017/best_accuracy.csv` 及 HKU `sharing_ablation_l2_20261019_v2`。[旧架构与配置](history/MODEL_ARCHITECTURES.md) · [共享/私有](history/SHARED_PRIVATE_MODEL.md) |

HKU 的旧版居中 8 天 MTLoRA 有提交和监视记录（`mtlora_centered8d_n20000_20261021`），按其后续 20k 对照的口径暂无独立成绩。另一次对历史 CGC 的站点留出评估在 CUDA 错误处中断（`cgc_heldout_station_0f8ab68/4483682.log`），该次未得到可报告的跨站结果。

冻结 patch embedding 的 MTLoRA 和固定门控三层循环 CGC 有配置，尚未核实可比的完整精度。早期 NLL、batch 一致性和梯度探查有诊断记录；扩容与时序的数项后续消融、CNN 专家权重迁移保留为方案。旧配置还包括 staged 独立/softmax 门控以及 LAI/LST 专项，运行状态应从各自实验目录补证。[历史诊断](history/MULTITASK_DIAGNOSTICS.md) · [SM 诊断与拟议消融](history/SM_CAUSAL_EXPERIMENT_PLAN.md) · [CNN 迁移方案](history/CNN_EXPERT_WEIGHT_SPACE_EXPERIMENT_PLAN.md)

## 数据、监督与区域试验

| 实验 | 得到的结论与适用范围 |
|---|---|
| 产品输入、标签与分层采样 | 经时空支持核对，PML 陆地 ET、水/冰 Ew 在纯像元上监督对应分量；混合像元的三个分量缺乏单独验证。SMAP 上午观测提供陆地 SM 粗支持。覆盖阈值小试、七个水体频率组、额外冰雪组和滚动候选池均有实采验收。成功配额按组计算；缓存复用时的首个成功时间与冷启动应分别解释。[PML 审计](PML_SUPERVISION_AUDIT_20261022.md) · [SM 调查](SOIL_MOISTURE_SITE_AUDIT_20261022.md) · [阈值小试](validation/RETRIEVAL_PILOT_20261022.md) · [分组](validation/RETRIEVAL_BALANCED_PILOT_20261023.md) · [冰雪](validation/RETRIEVAL_EIGHT_GROUP_PILOT_20261023.md) · [滚动候选池](validation/RETRIEVAL_ROLLING_POOL_20261023.md) |
| 站点时间、足迹与空间代表性 | 首批从 78,658 个已分层抽取的候选做空间检查与输入配对，候选数与成功数分开记录。SM 采用 ISMN 晨间观测和测点像元；ET 试验覆盖了足迹加权和无足迹的均质区域监督。后者使用 7×7 个 30 m 像元，分类有效覆盖率及同类占比均至少 90%，两项均存档。按任务隔离训练与留出站点，并核对站点留出与产品样本重叠。[试采](validation/SITE_PIPELINE_20261103.md) · [足迹验证](validation/SITE_FOOTPRINT_20261103.md) · [ET 全量质控](validation/SITE_ET_FULL_QC_20261104.md) · [划分](validation/SITE_INDEX_SPLIT_20261103.md) · [Zarr](validation/SITE_ZARR_PRODUCTION_20261104.md) |
| 站点 ET 闭合率、波文比校正、同站换日期补采、分层与产品对照 | 先试过按站点 EBR > 0.75 选原始 LE。112 个共同窗口上的 PML RMSE 在原始/校正 ET 下为 0.6077/1.3305 mm/day；按新时间质控重算为 111 个窗口，分别为 0.6079/1.3439。正式训练采用提供方波文比校正 LE、完整窗口实测时长至少 50%，取消整站 EBR 门槛与 30 天间隔。最终发布 217 个物理站点的 3,000 个 ET 窗口，训练/验证/测试为 2,108/446/446。其中 16 个按足迹聚合，2,984 个按均质区域监督，后者有 375 个的均质率在 90%–95%。水体频率为零的 2,915 个，(0,20%] 的 85 个来自 5 站；保留稀缺组优先和组内均衡，同一站点增加日期可以增加时间窗口，其空间独立性仍由站点数约束。[EBR 试验](validation/SITE_ET_EBR_PROTOCOL_20261110.md) · [初次配对](validation/SITE_ET_CORRECTION_PML_20261110.md) · [同窗重算](validation/SITE_ET_BOWEN_PILOT_20261110.md) · [空间规则与分层试验](validation/SITE_ET_BOWEN_RELEASE_20261110.md) · [发布与数量](validation/SITE_ET_TARGET3000_20261110.md) |
| 产品对站点、人工冰川点对产品、人工 Cover 接入 | 小样本开发清单中，SMAP 仅成功配对 937/1,639 条，PML 为 331/446 条。冰川人工点分层抽取 100 个，Dynamic World 同日有值 92 个、正确 78 个；项目 8 天高置信规则下保留 73 个、正确 71 个（水 34/34、冰雪 31/31、陆地 6/8）。提高有效子集的一致率也降低了覆盖率。早期八模型冰川外部点检的严格独立子集无水体，HKU 进度表记录 5 个模型完成全点评估；后续将冰川点与 Sen1Floods11 人工水体联合训练、分来源评估。抽查表在 `outputs/dw_points_100_20261123/pairs.csv`、`outputs/dw_points_100_8day/pairs.csv`。[站点产品配对](validation/SITE_PRODUCT_BASELINE_20261112.md) · [早期点检](validation/COVER_POINTS_20261121.md) · [人工标签协议](MANUAL_COVER_TRAINING.md) |
| 区域采集、推理和性能 | 鄱阳湖等区域做过 Drive 与原生像元下载对照、分块/重叠拼接与第二区域推理。相同缓存输入、同卡同 batch 的新旧推理比较中，七个输出逐值一致，有效掩膜相同，吞吐由 4.843 增至 5.865 窗口/秒。该数字适用于固定样本的模型前向，区域端到端还受读取和解压影响。另已用 shared-LoRA、MTLoRA 和全量微调权重生成水、冰、陆地和过渡带的验证地图，产物在 HKU `experiments/validation_maps_20261027`，用于空间误差诊断。[下载基准](validation/REGIONAL_PIXELS_BENCHMARK_20261025.md) · [区域推理](validation/REGIONAL_INFERENCE_20261029.md) · [第二区域](validation/REGION2_AND_INFERENCE_PROFILE_20261030.md) · [同输入性能复核](validation/REGIONAL_INFERENCE_SPEED_20261030.md) |

## 如何续记

- 新实验提交时登记研究问题、直接对照、唯一改变或一并改变的因素、数据发布/划分、预算/种子、代码提交及验收记录或运行目录。排队、启动、完整开发验证、完整测试分别标明。
- 完成后更新原条目，记录选模规则、选中轮次、按来源和变量分开的指标、有效样本数、关键差值和支持该解释的对照。失败、无提升和不支持原假设的结果一并保留。
- 详细逐轮表、复现配置和日志归入对应验证记录。本页保留查找入口、能回答的问题和会影响下一次实验设计的结论。复用相同测试集的后续迭代要注明，泛化结论优先用新种子、留出站点或独立事件复核。
