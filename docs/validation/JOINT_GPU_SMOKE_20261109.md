# 产品与站点联合训练 GPU 短测

2026-11-09，修复版本 `f4889e003c9680904173a6e375dad993a508b4f4`。
HPC 作业 `4410945`，`c_foss_gpu` / debug，单 NVIDIA L40、2 CPU。
最终状态 COMPLETED，退出码 0:0，总耗时 2 分 27 秒。

## 结果

- 完整 MTLoRA 模型，真实产品与两批站点数据，固定 v3 划分和训练统计。
- batch=2，累积=1；站点 SM/ET 损失权重均为 2。
- 训练、验证、测试各 16 条抽样记录（含重复代表样本），训练 8 步。
- 224 次任务头梯度有限性检查通过，28 个任务头参数张量发生改变。
- checkpoint 保存、重新加载及随后的 8 批测试全部完成。
- 修复覆盖正式 CLI 的显式本地加载及经同一方法调用的 resume；本次 GPU 短测
  验证了模型重新加载，未实际中断再恢复优化器训练。优化器状态往返由本地回归测试覆盖。
- 本地 checkpoint / 联合训练相关测试共 9 项通过。

首次作业 `4410807` 已完成训练和验证，但在加载含 HistoryBuffer 的 checkpoint 时失败。
原因是 MMEngine 未显式设置 torch.load 的 weights_only，而 PyTorch 2.7 默认限制
此类对象。HydroStateRunner 对可信本地 checkpoint 显式完整加载，远程加载规则及
PyTorch 全局设置保持不变。

## 证据与限制

结果目录：
`/scr/u/xiaoz/HydroState/experiments/joint_gpu_smoke_20261109_f4889e0/`

- `work/result.json`：通过状态、实际模型、梯度/参数更新与样本选择。
- `work/selection.json`、`work/smoke_config.py`：短测配置和索引。
- `work/epoch_1.pth`：保存并成功重新加载的 checkpoint。
- `slurm-4410945.out`：完整训练、验证与测试日志。

这不是正式精度评估，也没有完成最大 batch 探测。部分产品目标在代表子集中没有
有效标签，不能把该短测称为所有产品标签分支的全覆盖验收。
`result.json` 的 peak_allocated_gib 在 MMEngine 日志钩子重置峰值计数之后读取，
不是整个作业的峰值，不应用于 batch 规划；训练日志记录约 8965 MiB。
站点输入读取已完成的 FossFS 源数据，不依赖仍在发布的 SCR 目标目录。
