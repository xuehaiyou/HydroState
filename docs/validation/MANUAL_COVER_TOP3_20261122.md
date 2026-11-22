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
