# CoSA 联合训练剩余 GPU 验收 — 2026-11-10

## 修复依据

前次 L40S debug 4414518 耗时 7 分 43 秒，FAILED 1:0，未触发正式训练。
没有发生 OOM：C 的 batch 16/20/24/28 短测、B 的 batch 24 八步训练均返回 passed。
B 的实际张量峰值 29.784 GiB，缓存池峰值 42.449 GiB，设备可见容量 44.392 GiB；
控制脚本把缓存池占用也要求限制在容量 90% 以下，因而中止。

PyTorch 的 allocated 是张量占用，reserved 包含可复用缓存，不能当作相同含义。
依据：[PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.7/notes/cuda.html#memory-management)。
新的共同判据是 `peak_allocated + peak_external <= 0.9 * device_total`；
external 在同步采样点按 `total - free - current_reserved` 测量并取非负峰值，
涵盖 CUDA context、库等 allocator 外占用。reserved 继续报告，不作为单独失败条件。
这不是 external 连续峰值的保证；仍需真实多步测试、OOM 检查及 10% 余量。
不在迭代中清空缓存，也不改变正式训练 allocator 或模型。

## 本次测试内容

- `gpu_memory.py` 统一 benchmark、完整 smoke 和正式提交判据，跨日志重置保留峰值。
- `tools/experiments/debug_joint.py` 在同一 debug 作业内预检真实批次，集中探测
  batch 24/28/32/36/40，遇到 OOM 或余量不足停止上探。
- 从最大通过候选开始，B/C 各八步真实混合数据训练吞吐测量及三批验证，
  再各四步完整 Runner 训练、代表性验证、checkpoint 保存/恢复和抽样测试。
- 完整测试若发现 OOM/余量不足，保留证据并尝试下一档已通过候选；其他数值、
  数据或 checkpoint 错误直接失败，不用降低 batch 掩盖。
- 保存后清零所有可训练参数，再严格加载并逐项精确比较；随后实际运行测试集子集。
  本次 GPU 检查证明模型参数重载，优化器恢复由已有本地回归覆盖，未模拟 GPU 中断续训。
- 两配置必须使用相同最终 micro-batch、累积 1；继续使用原始 FM 初始化，
  产品/站点各半、站点损失系数 2、站点选模权重 2/3、各 50 epochs。

仅两套验收报告全部通过、Slurm debug 完成且退出 0:0，后台控制器才提交正式 B/C。
正式任务沿用单 L40S、4 CPU、2 loader workers、SCR 数据；按实测耗时加 10%，
超出七天按 checkpoint 边界续跑。代表性子集的精度不作为正式验证结论。

## 第三次 debug：通过显存测试，发现批量元数据错误

本次相关本地测试共 22 项通过：显存采样/判据、严格参数恢复、失败批量回退、
提交门槛/去重、既有 checkpoint 与分段续跑。Ruff 和差异空白检查通过。
模型与测试源码提交 `da5dad7fbb833312410b678492eb5397dd187c6f`，已推送 GitHub、
同步 HPC 并冻结 release。作业 **4416920** 于 11:41:31 HKT 在 SPG-5-4 启动，
单 L40S、4 CPU、48 GiB 主存、debug 25 分钟；最终 FAILED 1:0，耗时 12 分 22 秒。
真实混合批次预检、batch 24/28/32 四步探测和 B/C 的 batch 28 八步训练均完成。
batch 32 跑通但张量峰值 39.510 GiB 加外部 0.529 GiB 超过 39.953 GiB 门槛，
故选择 batch 28 进入完整验收。B 保存 epoch checkpoint 后，在验证指标处失败：
批量数值元数据 `year` 没有逐样本拆分，`int(batch_tensor)` 报错。
控制器已停止，正式训练未提交。

运行根目录：`/scr/u/xiaoz/HydroState/experiments/cosa_joint_l40s_20261110_r3`。
`debug_passed.json` 仅在全部通过后生成；候选试验保存在 `batch_N/`，根目录的
`benchmark_b/c.json` 和 `smoke_b/c` 指向最终通过的同一批量证据。
自动控制器 PID 621923；最终状态查 `submission.json`、`controller_status.json`、
`controller_failure.json` 和 Slurm，不能仅凭控制器进程存在判为成功。

## 元数据与代表性监督补充修复

`HydroStateModel.predict` 现在把批量张量元数据与字符串列表一样按样本拆分。
在真实 Zarr 混合批次的本地回归中，用不同年份并启用共同年份指标，修复前复现同一异常。
修复不改变模型前向、损失、输入数据或年份筛选定义。

同时核对有效监督覆盖：manifest 的计数键为 `et_ice_native_valid` 等实际掩膜名，
旧短测选样错误地按任务名查询，遗漏最高有效监督的代表样本。现在从任务规格派生掩膜名，
并要求完整 smoke 中七个产品任务及两项站点任务的有效监督计数都大于零。
原始训练统计中冰雪 ET 有 3,124 个有效原生监督值，不是数据集缺少该变量。
这个修复仅改变验收子集，正式训练的采样、划分和统计不变。

修复后重试用新目录 `cosa_joint_l40s_20261110_r4`，根据已有显存结果从 batch 28 开始，
重新检查含新增代表样本的批次及 B/C 完整流程；旧目录和失败证据保留。
新增修复相关的联合数据、模型预测、CoSA、验收检查共 47 项本地测试通过；Ruff 和差异检查通过。
