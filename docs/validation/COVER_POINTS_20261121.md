# 人工覆盖点接入与运行记录（2026-11-21）

## 已实现与测试

实现说明见 [COVER_POINT_VALIDATION](../COVER_POINT_VALIDATION.md)。
生产代码 commit `46f7ec573774efa4defd60159b8206bd0c41257b`；
显存探测/短测代码 `f5e3dd2`；完整评分及自动提交控制器代码 `3426445`。
三版本均已 Git 推送，并在 HPC 独立 detached worktree 检出，未覆盖运行中的训练代码。

本地 9 项测试通过，Ruff 与差异检查通过。测试包含真实 tiny OlmoEarth 无标签预测路径，
不代表正式 GPU checkpoint 的实际推理已通过。

## 数据

`/scr/u/xiaoz/HydroState/samples/glacier_cover_v1`

11,729 个人工点、378 个空间—时间窗口、4 处冰川、8 个地点—日期组合。
六个目标周期均与既有产品缓存中的实际 PML 日历吻合。
类别：冰雪 7,643、岩石/碎屑陆地 3,937、水 149。
900 m 所有权核心 + 四周 150 m 上下文，输入为 1200 m；每点只有一个评分窗口。

6 个代表性窗口实采及完整数组/标签映射审计通过：1,582 个点均有输入支持，
其中冰雪 1,098、陆地 435、水 49，覆盖全部四处冰川。
全标签表有 14 个同日同输出像元异类冲突，保留并报告，不静默多数投票。

全量采集在 hpc2021-io2 上运行，4 并发、绑定 2 CPU、代理已禁用；
进程启动记录为数据目录的 `collection_process.json`，日志 `collection.log`。
2026-11-21 23:29 HKT 检查时成功 85/378 窗口、拒绝 0，全量尚未完成。
成功后自动执行全量 audit 和发布；不应把试采审计视作全量发布完成。

## 训练重叠与结论边界

对下述八模型实际选择的 4,528 个训练窗口分别审计，保守排除输入外接圆及 35 km
邻近范围；结果均为 51/378 个验证窗口落入排除范围，涉及 SouthCascade 的 2,612 个点。
严格子集余 9,117 个点（冰雪 6,006、陆地 3,111），其中没有水体点。

因此完整报告包含“全部有效点（含训练邻近范围）”和“独立子集”两套指标。
全部 149 个水体点属于前者，不能给出独立水体精度结论。
这仍是保守邻近筛选，不是证明每个人工点本身已经作为训练标签出现。

## 模型与作业

模型清单：`/scr/u/xiaoz/HydroState/experiments/cover_validation_20261121/models.json`。
采用 `architecture_experts_20261118_v2` 的既有站点选模权重：
A、B、CGC、CGC_matched、PLE、MLoRE_complete、CoSA_Router、AdapterFusion。
全部为其 `best_site_score_epoch_6.pth`；MLoRE 使用 `retry_batch2/work`。
未以人工 Cover 分数重新挑选 checkpoint，未中断或修改其他运行中的训练。

短测 Slurm 作业 `4485195`，l40s/debug，单 GPU、4 CPU、64 GB、20 分钟；
2026-11-21 23:29 HKT 尚在等待资源，没有正式 GPU 验收结论。
短测会分别测量 batch 上限、保留 15% allocated 显存余量，并做六窗口端到端推理。

控制器在登录节点轻量等待（记录 `controller_process.json`）；
只有数据发布且短测作业成功后，才按实测速度与初始化耗时加 10% 余量提交完整评估。
原运行目录下 `controller_status.json`、`full_submission.json` 为后续状态来源。
最终报告预定为 `suite/common_metrics.json`，当前尚未生成。
