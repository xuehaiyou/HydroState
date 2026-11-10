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

## 执行状态

本次相关本地测试共 22 项通过：显存采样/判据、严格参数恢复、失败批量回退、
提交门槛/去重、既有 checkpoint 与分段续跑。Ruff 和差异空白检查通过。
尚未提交本次 GPU debug；后续补充实际 commit、作业及结果。
