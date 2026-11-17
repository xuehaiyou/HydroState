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
