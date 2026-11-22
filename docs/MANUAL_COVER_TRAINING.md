# 人工 Cover 比例样本与三架构联合训练

2026-11-22 用户确认：只训练上一轮站点 `site_score` 前三名 CGC、CoSA_Router、B。
冰川与洪水样本合并打乱、每 epoch 无放回遍历；不设来源损失系数或采样倍数。
旧 checkpoint 不续接：从原架构预训练主干初始化，任务模块重新初始化。
原产品和站点监督保留；两种人工标签使用同一个独立比例 MSE，系数 2。
每窗口先对各变量的有效像元平均，再对有效变量平均；不同来源不另行加权。
洪水只有 water 有效，不能生成 snow_ice/land 的独立标签，亦不重复计入产品 NLL。

## 样本契约

新根目录 `/scr/u/xiaoz/HydroState/samples/manual_cover_v2/`，旧发布不变。
契约 `hydrostate_manual_cover_v2`，输入仍为 landsat_l1_toa_native_mean_v2：
1200 m 窗口，5×8 天，原生波段存储，读取 120×120，输出 40×40。
冰川复用旧版已发布输入，由清单显式记录旧输入契约及来源哈希；标签使用新契约。
这是版本化引用，不复制旧输入或修改其元数据。

- 冰川点：用户选择将所在 30 m 像元近似为纯类，三类 one-hot；不是实测亚像元比例。
  同像元同类点合并，异类冲突不生成比例监督；原点表保留。无点位置掩膜无效。
- Sen1Floods11：只用 `HandLabeled/LabelHand`，不用 S1Otsu 或弱标签。
  原值 -1/0/1 分别是无效/非水/水。原始像元多边形变换到目标 UTM 后按重叠面积聚合。
  重叠源片先取几何并集，异类重叠区域剔除。有效面积不足完整输出像元的不参与主监督。
  发布水体比例、有效面积比例及独立有效掩膜，不推断陆地/冰雪比例。
- 来源保留 S1/S2 日期；人工标注使用多传感器信息，以 S2 日期锚定目标周期，S1 日期另存。
  单日人工标签对应 8 天状态是一项明确近似。Mekong 芯片对应元数据 Cambodia 事件。
  人工标签不按产品采样的 2017–2024 年裁切；India 2016 年 68 片纳入，
  能否使用由真实日期对应的输入覆盖和质量决定。现有目录通过有记录的增量补充保留已完成样本。
- 原始标签文件只比较大小用于传输，不额外逐文件计算校验和；发布元数据另有版本身份绑定。

标签先生成独立 `labels/<id>.zarr`，`_LABELS_READY.json` 不是完整样本发布。
只有输入实采、空间排除和全量审计通过才写 `_PUBLISHED.json`。
`inventory.parquet`、`cohort_<source>_<split>.parquet` 与 samples.parquet 在该版本目录内发布。

## 固定空间划分与隔离

冰川：Gulkana、Wolverine 训练；Sperry 验证；SouthCascade 测试。
原始来源的 site_name 可能在同一窗口包含多个站名；按实际坐标分组，原始点和站名不改写。
洪水整事件保持同组，跨事件/来源的邻近支持先合并。60/20/20 是窗口量软目标。
跨集合要求区域包围支持间隔超过 35 km；同时输出实际原训练支持距离和排除清单。
原产品的 SMAP 支持用现有支持几何，站点输入包围圆另加 35 km 保守范围。

未来训练清单：排除靠近人工验证/测试的原训练样本；排除靠近原验证/测试的人工训练样本；
两侧验证样本还排除另一来源测试支持邻近区。排除只发生在新清单，不改写原始数据。
产品和站点标准化统计量使用实际保留的训练样本重新计算。
已有 checkpoint 的历史接触不会被新划分消除；旧冰川点已查看过，不能称为盲测。

## 训练与选模

保留上一轮 10% 产品/站点训练子集和 20% 开发验证选择作为起点，应用新空间排除后，
加入全部有效人工训练窗口；测试使用固定原测试集和人工测试集。
6 epochs，每 epoch 验证，AdamW lr=1e-4、weight_decay=0.01，BF16，梯度裁剪 1。
有效 batch=10；debug 实测 micro-batch 5，不满足显存余量则试 2，累积分别为 2/5。
100 次优化器更新 warmup、6 epoch cosine。验证/测试 batch=8，不使用此前的 batch=64。

冰川、洪水单独验证；冰川三类比例误差与分类精度，洪水只有水体比例误差及二分类水体指标。
缺少真值支持的类别不报告虚假零精度；两套来源不能拼成一个三分类总体精度。
`manual_joint_score = -(cover_MAE + SM_RMSE/std_train_SM + ET_RMSE/std_train_ET)/3`。
cover_MAE 为冰川三类平均 MAE 与洪水 WF MAE 的等权平均；这是选模规则，不是来源训练权重。
站点 SM/ET 原始指标和产品一致性指标同时保留；测试只在验证选定 checkpoint 后运行。

## 入口和自动提交

```
python tools/data/manual_cover.py prepare --root ROOT --glacier-root OLD_ROOT --flood-root RAW_V1_1
python tools/data/manual_cover.py labels --root ROOT
python tools/data/manual_cover.py audit-training --root ROOT --manifests PRODUCT --manifests SITE_INVENTORY
python tools/data/manual_cover.py collect --root ROOT --project PROJECT --pilot
python tools/data/manual_cover.py audit --root ROOT --pilot
python tools/data/manual_cover.py collect --root ROOT --project PROJECT
python tools/data/manual_cover.py audit --root ROOT --publish
python tools/experiments/launch_manual_cover.py run --root ROOT --models PREVIOUS_MODELS_JSON --output RUN_ROOT
```

数据阶段使用 IO 节点 2 CPU、4 并发并禁用代理变量。控制器在完整发布前不申请训练 GPU。
发布后重建清单与统计量，再提交三个 l40s/debug 单卡验收；通过后按实测速度估算正式预算，
保守计入 I/O 差异并加 10% 余量，必要时按 epoch 分段、保存 optimizer/scheduler 后续训。
提交意图和回执防止重复；任一验收失败不提交正式训练。所有生产和训练使用独立 Git release。

## 原始来源

- https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md
- https://github.com/cloudtostreet/Sen1Floods11/blob/master/README.md
- https://tc.copernicus.org/articles/19/1675/2025/
- https://github.com/RaineyAbe/glacier-snow-cover-mapping

### 三候选结果汇总

```bash
python tools/experiments/summarize_manual_cover.py TRAINING_ROOT --output REPORT.json
```

汇总固定 CGC、CoSA_Router、B。每个模型按 `manual_joint_score` 选择同一验证 epoch，
分别保留该 epoch 的 Cover、SM、ET 指标和选定 checkpoint 的最终测试指标；
不会拼接各任务在不同 epoch 的最优结果。训练不足六轮、缺少 checkpoint 或最终测试
结果时标为不完整。指标可用仍需结合 Slurm 退出状态核实成功，不能仅凭日志判断任务完成。

## 后续架构的比较与迭代

首轮在相同训练、验证清单上运行 CGC、CoSA_Router、B。后续新结构沿用同一人工 Cover 发布和
空间排除清单，使用相同初始化规则、训练统计量、有效 batch、种子和更新预算。开发阶段运行
`python tools/experiments/summarize_manual_cover.py TRAINING_ROOT --validation-only`，
只读取开发验证结果。按既有 `manual_joint_score` 为每个模型选一个 epoch，使用其未舍入指标进行横向比较。

| 观测来源 | 主要误差（越低越好） | 同时检查 |
|---|---|---|
| 冰川人工点 | 水、冰雪、陆地三个比例 MAE 的算术平均 | 各比例 MAE、三分类 OA、各类支持数和混淆矩阵 |
| 洪水 LabelHand | 水体比例 MAE | 水体 IoU、F1、二分类混淆矩阵，按事件列出 |
| 站点 SM | 原单位 RMSE（m³/m³） | Pearson R、站点等权 RMSE、站内去均值 R |
| 站点 ET | 原单位 RMSE（mm/day） | Pearson R、站点等权 RMSE、站内去均值 R |

汇总工具输出各指标的领先者、同时取得四项最低误差的交集，以及在三类冰川 MAE、冰川 OA、
洪水水体 IoU 和两个站点 R 上也同时领先的交集。平局保留并列。有效数量不同则先查逐条配对；
完成六轮和检查点选定前不做最终横比。站点等权与站内异常结果需由逐观测预测计算；
各开发模型比较使用相同有效的站点窗口、掩膜和空间支持。

每轮只改变一个可解释的结构决定。先记录假设、对照和预期出现变化的指标，再用同一模型权重关闭
或固定该路径，区分它在推理时的作用与重训的间接作用。若出现任务权衡，记录具体受益和受损的
来源及地区；没有证据支持的路径不继续堆叠。能在开发集同时改善多项指标的结构与最强基线再用
相同的新种子复验，比较逐站、逐事件的配对差异。冰川验证集中在 Sperry，洪水验证只有两个事件；
像元数无法提供同等数量的独立地区证据。最终冻结结构、种子与检查点选择规则，再解读留出测试结果。
已查看过的测试地区要注明评估历史。没有独立观测覆盖的 ET 分量仅报告产品一致性。
