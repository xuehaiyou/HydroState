# 人工 Cover 发布与前三架构训练运行记录

截至 2026-11-22 15:39 HKT：标签转换 2875/9726 窗口；尚未完成输入采集，未发布完整样本，
尚未提交 GPU 训练。训练控制器状态 `waiting_for_published_data`。
正式方案见 [人工 Cover 联合训练](../MANUAL_COVER_TRAINING.md)。

## 代码与验收

- 初始数据实现 `0597017`；Mekong/Cambodia 元数据显式映射修复 `26ee9d6`。
- 统一人工比例损失、混合数据读取、选模及提交器 `0262f99`。
- 分类准确率和混淆矩阵输出 `a549344`；最终测试禁用 optimizer resume `f63c65f`。
- 数据生产 release：`/home/xiaoz/HydroState-releases/manual_cover_26ee9d6`。
- 训练 release：`/home/xiaoz/HydroState-releases/manual_train_f63c65f`。
- 相关 63 项测试通过（比例面积、无效区、空间分组、混合拼批、真实模型梯度、原站点/产品路径）。
  随后的最终测试加载检查通过；Ruff 和差异检查通过。GPU 验收尚未运行。

## 数据和流程

数据根 `/scr/u/xiaoz/HydroState/samples/manual_cover_v2`。
运行根 `/scr/u/xiaoz/HydroState/experiments/manual_cover_top3_20261122`。
446 幅人工源片中纳入 378 幅，排除 India 2016 年 68 幅；来源保留在 SCR sources。
只传输约 3.3 MB 人工标签与事件元数据，rsync 使用 size-only；GEE 进程清除代理变量。
冰川 378 输入窗口直接引用旧发布；总计划 9726 窗口，含没有完整有效标签的候选边缘窗口，
最终成功样本数必须以完成标签筛选、输入采集及发布审计后的清单为准。

初步事件分组：洪水训练 Mekong/Nigeria/Paraguay/Sri-Lanka/USA；
验证 Bolivia/Ghana；测试 Pakistan/Somalia/Spain。
冰川按文档中的地区安排，真实坐标邻近分组优先于名称。

IO2 标签转换进程和日志：`preparation_retry_process.json`、`preparation_retry.log`。
此前元数据别名失败的未发布目录保留为 `manual_cover_v2_failed_metadata_0597017`。
IO2 后续数据控制器 PID 3797868，`data_controller.py`、`data_controller.log`：
等待标签完成 → 现有训练支持审计 → 代表性实采与审计 → 全量采集 → 发布。
任一步失败写 `data_failed.json`，不冒充发布或自动启动训练。

登录节点训练控制器 PID 682409，`training_controller_process.json`、`training_controller.log`：
等待完整发布 → 重建空间隔离清单和训练统计量 → 三个 debug 验收 → 正式训练与最终测试。
状态为 `training/status.json`，实际 Slurm job ID 将写入各阶段 `submission.json`。
每卡 4 CPU，l40s/debug 验收 20 分钟（已核实 QoS 上限 30 分钟）；正式 walltime 按实测预算。
当前没有这三组新训练的 Slurm job ID，不能把控制器 PID 称为训练作业号。

## 用户取消的旧任务

用户明确要求取消旧 `hs_cover_resume32`，作业 4487155 已从 PENDING 取消，sacct 确认 CANCELLED。
这是旧 checkpoint 的冰川人工点验证续跑，与本轮三模型重训独立；已有结果保留。

## 2026-11-22 人工标签年份范围补充

用户确认在当前 `manual_cover_v2` 目录增量补充，不创建新数据版本。此前将产品
2017–2024 年范围用于人工标签准入是不必要的限制。代码提交 `9c1de94` 移除了
Sen1Floods11 和冰川点导入的固定年份检查；输入仍按真实观测日期和质量门槛采集。

实际执行的补充结果：

- India 2016 年 68 幅 LabelHand 已纳入清单，新增 1,657 个计划窗口；
  总计划从 9,726 增至 11,383。全部 446 幅人工洪水源片纳入候选。
- 新增 India 分配到 train；按区域包络支持检查与原 val/test 的 35 km 间隔通过。
  原有窗口 ID、划分、标签和输入生产配置均保留。
- 冰川原始 CSV 的 11,729 个点均为 2021-06-15 至 2021-08-29，逐行比对
  source_row、位置、日期和原类别与现有点表一致，无遗漏，无需新增冰川窗口。
- 补充前清单、审计和哈希保存在数据目录 `amendments/remove_year_cutoff/`。
  `report.json` 绑定补充脚本哈希、原始清单哈希和新计划哈希。
- 像素生产继续使用冻结提交 `26ee9d6`，补充脚本来自 `9c1de94`；修改仅涉及
  源片选择，未重写原生产指纹。后续新建数据可直接使用取消年份限制的新导入代码。

旧 IO2 数据控制器及其采集子进程已停止；已完成接收记录保留。IO2 新控制器 PID
4115094 已启动，先转换新增标签，再重做训练支持审计、包含 India 的代表窗口实采、
全量断点续采和发布审计。采集保持 2 CPU、4 并发并清除代理环境变量。
三个候选模型仍由原训练控制器等待完整发布后启动。此记录不代表新增输入或训练已完成。

本地验证：17 项相关测试通过，覆盖 2016/2025 年点导入、2016 年洪水导入与目标时间、
新增窗口保留旧划分以及拒绝与 holdout 过近的区域；Ruff 和差异检查通过。

补充标签转换已完成（`_LABELS_READY.json`，耗时 1,022.97 秒）：全部 11,383 个窗口
生成标签，10,375 个有有效监督；India 新增 1,657 个中 1,561 个有效、96 个无完整
有效标签像元。原有 9,726 个窗口的 ID、split、spatial_group、target_time、transform
与补充前备份逐行比对一致。此时仅标签就绪，输入采集及最终发布仍未完成。

| 来源 | 划分 | 计划窗口 | 有效标签窗口 |
|---|---|---:|---:|
| glacier | train | 289 | 289 |
| glacier | val | 19 | 19 |
| glacier | test | 70 | 70 |
| sen1floods11 | train | 7,272 | 6,866 |
| sen1floods11 | val | 1,854 | 1,401 |
| sen1floods11 | test | 1,879 | 1,730 |

以上是标签窗口统计，后续还需输入可用性筛选和相对历史站点/产品 holdout 的训练隔离，
不能直接作为最终训练/验证样本量。
