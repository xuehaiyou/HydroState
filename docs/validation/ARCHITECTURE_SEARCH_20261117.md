# 少样本架构搜索运行记录

## 已核实

- 选择清单：`/scr/u/xiaoz/HydroState/experiments/architecture_search_20261117/selection_v2`。
- 子集统计：同目录 `stats_10`，CPU 作业 `4462371` 成功完成，耗时 2 分 37 秒。
- 训练窗口共 4528：产品 1413，站点 3115。统计量仅来自这一训练子集。
- 站点 ET 开发验证保留完整 446 窗口，其余按固定开发子集筛选。
- A（shared-LoRA）和 B（CoSA independent）已通过真实清单与统计量契约检查。
- 代码 `5e0a700fab4a6808b8a3dc105dd7882902a80be4`；相关本地测试 21 项通过。

## GPU 验收

作业 `4462420`，`c_foss_gpu` / `debug`，单 GPU、4 CPU、48 GB 主存、22 分钟。
提交后核实为 PENDING / Resources，尚未声称 GPU 测试通过。
先探测 A/B 的匹配 batch（4、6、8），再执行训练、开发验证和 checkpoint 重载检查。
重载检查的 test API 显式绑定开发验证 loader/evaluator，**不读取封存测试划分**；
脚本中历史 `test_joint_score` 字段仅是验收诊断，不能作为测试精度报告。
产物：上述实验根目录 `debug_v1`，提交回执 `receipt_foss.json`。

集群拒绝了 `l40s,c_foss_gpu` 多分区提交（Invalid partition name）；该次未创建任务。
改为单分区提交后获得上述唯一作业号，没有重复运行。

## 后续门槛

GPU 验收通过后，以实测吞吐、验证耗时和初始化开销确定 6 epoch 筛选预算及 10% 时间余量。
短测只证明执行与显存可行，不证明架构精度；尚未启动本轮 A/B 少样本正式训练。
已有 E 作业 `4462162` 保持运行，不因搜索流程改变而中断。

## GPU 短测更新

`4462420` 已 FAILED：L40 上外部进程占用约 16.13 GiB，模型分配约 26.93 GiB
时 OOM。不能用这一受外部占用影响的结果估算空闲单卡的最大 batch。
尝试从登录节点 SSH 到计算节点核对 PID 归属时主机名解析失败，归属尚未确认，
没有停止其他进程。诊断保存在 `debug_v1/A_4.json`。
已在另一允许分区 l40s/debug 提交 `4462428`，产物为 `debug_v2`；新增记录
CUDA/Slurm GPU 绑定、UUID、显存及计算进程的启动诊断。没有重启已有正式训练。

## L40S 验收结果（22:50 核实）

`4462428` 已完成，两模型 batch 4 的真实数据验收均 passed。

| 模型 | 训练窗口/秒 | 验证前向秒/窗口 | 峰值分配显存 GiB | 重载核对参数数 |
|---|---:|---:|---:|---:|
| Shared-LoRA A | 2.490 | 0.1305 | 28.147 | 449 |
| CoSA independent B | 3.627 | 0.1141 | 23.908 | 301 |

这是短测，不是全 epoch 吞吐；验证前向时间不包含完整指标计算与存盘开销。
两模型验收均触达全部七项产品任务及站点 SM/ET，有有限梯度与权重更新。
各自完整 smoke 耗时约 84/74 秒，含代表性开发验证及 checkpoint 重载。
batch 6 的 A 运行成功，但显存未满足 10% 余量，因此尚不采用。

补测 `4462451` 已提交 l40s/debug（单 GPU、4 CPU、14 分钟），代码 `0fd57eb`。
任务探测 batch 5，并对 A/B/CGC/MLoRE 进行同口径验收；如 batch 5 显存不足则测 batch 4。
结果存于 `debug_v3`，与前轮输出分开，依赖前轮 afterok，提交时没有重复计算作业。
完整测试集仍未读取；新模块只有 CPU 验证证据，尚未声称 GPU 验收或精度通过。

## 四模型 GPU 验收与筛选提交完成

`4462451` COMPLETED / 0:0，耗时 8 分 35 秒。A/B/CGC/MLoRE 的 batch 5
基准和完整 smoke 均通过，全部七项产品任务及站点 SM/ET 均有实际有效监督，
有限梯度、权重更新、开发验证、checkpoint 重载通过。仍不构成精度优势证据。

自动提交任务 `4462460` COMPLETED / 0:0，耗时 28 秒；控制代码 `9fc7bde`。
以下训练均使用科学实现版本 `0fd57eba8a11ea8245d5e5ad9351fdbfa3e99bd2`，
batch 5、梯度累积 1、6 epoch、每 2 epoch 开发验证一次、固定 4528 训练窗口。

| 模型 | 作业 | 申请时限（含 10% 余量） | 顺序依赖 |
|---|---|---|---|
| Shared-LoRA A | 4462483 | 04:06:00 | 无 |
| CoSA independent B | 4462484 | 03:04:00 | afterok A |
| CGC | 4462485 | 03:05:00 | afterok B |
| MLoRE block | 4462486 | 03:08:00 | afterok CGC |

均为 l40s/gpu、单卡、4 CPU、48 GB 主存；一次只运行这一组中的一个模型。
时限来自各自实测吞吐及 smoke 运行开销，加 10% 后向上取整到分钟。
首次核实 A 为 PENDING/Priority，其余为 PENDING/Dependency，没有声称已经训练。
配置、脚本和提交回执在实验根目录 `screen_10/<模型名>/`，汇总为 `screen_10/plan.json`。
训练重新从 FM 预训练权重和新任务参数出发，不加载短测 checkpoint；不自动运行封存测试集。
