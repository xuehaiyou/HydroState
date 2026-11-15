# AdapterFusion 实验 D：2026-11-15 GPU 验收与正式提交

## 固定来源

- 正式运行代码：`1d968cb04a61a2f604e94dcb9b534cad826dcfca`。
- 模型配置：`configs/hydrostate_joint_adapterfusion.py`；架构见 [说明](../ADAPTERFUSION.md)。
- 官方 BertFusion 类与固定上游 commit 的对应代码逐字一致，初始化 helper 也来自上游。
- A 初始化：shared-LoRA 产品＋站点联合训练的 `best_joint_score_epoch_8.pth`；
  SHA256 `c2d1546448c49153f07c933499c858461708f8d46c64cf89b90f67a9490883f4`。
- B/C 对照也固定各自当前综合最佳 epoch 8；绝不读取后续变化中的 best 文件。
- 数据沿用 `retrieval_eight_20000_06aba36` 与
  `site_et_bowen_target3000_v3_20261110` 发布版本；44,514 个训练窗口、8,751 个验证窗口。

## 已通过

- 本地 33 项测试（AdapterFusion、MTLoRA 与 retrieval model）；相关 Ruff 与差异检查通过。
- Debug `4441562`：真实输入零残差预测与基线最大绝对差为 0，融合参数能更新。
  初始验收从 22 个候选条目中检查了 12 个条目（前三批）；旧验收文件的 samples=22
  表示候选数量。后续代码明确区分 candidate_samples/tested_samples。
- Debug `4441582`：完整 MMEngine 训练、验证、保存 checkpoint、严格重新加载通过；
  训练后逐项确认所有原模型 state_dict 张量与基线完全一致。
  完整训练器短测使用全部 22 个代表性条目，包含产品与站点来源。
- 总参数 118,329,088；可训练参数 **86,919**，仅融合模块和残差系数。
- batch 8/16/32/64 均通过；80、96 出现 CUDA OOM。OOM 是容量探测结果，不是正式训练失败。
- batch 64：峰值张量 31.49 GiB，外部 CUDA 等占用约 0.53 GiB；设备可用总量 44.39 GiB。
  分配器保留缓存约 43.60 GiB，不等于全部为活跃张量。
  含读取的训练吞吐约 5.56 窗口/秒，验证约 6.41 窗口/秒；短测速度不代表长期保证。

## 正式作业

- 作业 **4441599**，`hs_adapterfusion_10`，L40S 单卡、4 CPU、48 GiB 主存、2 workers。
- batch=64，累积=1，有效 batch=64；不同于原 A/B/C 的 batch=4，需要作为阶段差异报告。
- 10 epoch，所有梯度 checkpoint 关闭，按原 joint_score 选模。
- 预计训练、逐轮验证及最终共同子集比较总耗时 26.72 小时，加 10% 余量后申请 29:24:00。
- 已提交并进入 RUNNING；本记录不代表训练完成，也没有正式精度结论。
- 原三个训练任务未修改。

HPC 运行根目录：
`/scr/u/xiaoz/HydroState/experiments/adapterfusion_joint_20261115`

记录包含 baseline.json、comparison_models.json、formal_config.py、formal_receipt.json、
GPU 验收和容量报告、源码/数据指纹；训练日志在 `logs/formal_4441599.out/.err`，checkpoint 在 `work/`。

正式训练成功后，同一作业自动评估 A/B/C/D 与产品的共同有效站点子集，输出 `comparison.json`。
复用既有产品配对值，不重复下载；评估使用既有验证过的 batch=4。
D 使用融合训练的综合最佳 checkpoint，A/B/C 使用上述固定 epoch 8；
这是额外融合训练实验，不能宣称与 A/B/C 从头训练具有相同累计预算。
