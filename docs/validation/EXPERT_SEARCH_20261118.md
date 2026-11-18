# 完整专家架构与站点选模筛选（2026-11-18）

用户授权实现并在 HPC 运行：完整 MLoRE、多层 PLE、匹配 CGC、AdapterFusion、CoSA Router。
A/B/原 CGC 一同复验，用于统一有效 batch 与新的站点选模口径，共八组。

## 科学设置

- 沿用 architecture_search_20261117/selection_v2 和 stats_10：4528 训练窗口、2261 开发窗口。
- 各模型六轮、每两轮开发验证，FM 预训练初始化＋新增任务模块从头训练，不加载旧任务 checkpoint。
- 有效 batch=10，GPU 短测在其约数 1/2/5/10 中选择最大安全 micro-batch，梯度累积对齐。
  相对前轮有效 batch=5 有改变，故重新训练基准；不同 micro-batch 的 BatchNorm 统计仍有差异。
- 只按 `site_score = -(RMSE_SM/std_train_SM + RMSE_ET/std_train_ET)/2` 选模，两个目标必须均有观测。
  保留 Pearson R、RMSE、MAE、Bias 和产品一致性。产品不参与该分数。
- 短测 test API 显式绑定开发验证 loader/evaluator，仅作 checkpoint 重载检查；不读取封存测试集。
- 完整 MLoRE 使用官方 ViT-B 解码配置及负载均衡损失；详见[来源与差异](../../hydrostate/models/mlore/PROVENANCE.md)。
- PLE 的层级、门控和空间接口见 [PLE](../PLE.md)。原单层 CGC、单层 MLoRE 不覆盖。

## 验证与运行

本地已通过完整解码路径与固定官方 forward 的 FP32 数值对照、负载均衡公式、
PLE 共享门控解析结果和跨任务梯度、缺测掩膜、真实小型编码器的混合监督，以及
站点选模不受产品误差影响的测试。HPC 任务号和验收结果在提交后追加。

提交入口 `tools/experiments/expert_search.py`。每候选使用独立 GPU debug 作业探测、
前后向及 checkpoint 重载；通过验收后依据实际吞吐、历史短测到端到端的校准系数，
加 10% 时间余量提交训练。候选训练按 afterany 顺序依赖，某模型失败不阻止其他模型。
已有运行不会被中止或覆盖。

## HPC 提交（12:57 HKT 核实）

冻结实现 commit：`c6f9198562364b32dd267aaea2cad4fac7315a79`。
代码由 Git 同步后在 SCR 创建独立 release，不修改既有训练的冻结目录。
实验根目录：`/scr/u/xiaoz/HydroState/experiments/architecture_experts_20261118`。

| GPU 验收 | Slurm job |
|---|---:|
| 完整 MLoRE | 4464680 |
| PLE | 4464681 |
| 匹配 CGC | 4464682 |
| AdapterFusion E | 4464683 |
| CoSA Router | 4464684 |
| A／B／原 CGC 合并验收 | 4464685 |

均为 l40s/debug，单 GPU、4 CPU、64 GiB 主存、20 分钟上限，按 afterany 顺序执行。
首次 B 单独提交被 `QOSMaxSubmitJobPerUserLimit` 拒绝，未创建作业；已核实 debug
最多同时提交 6 个任务，将 A/B/CGC 合并到仍 PENDING 的 4464685 启动脚本。
拒绝回执保留，三个基准各有独立输出与验收报告。

自动训练提交作业 **4464693**：c_foss_amd/normal、1 CPU、4 GiB、3 分钟，
依赖最后一个验收作业结束。它逐项要求验收作业 COMPLETED/0:0、显存余量及
checkpoint 重载证据，才提交对应训练；回执防止重复提交。正式训练使用
l40s/gpu、单 GPU、4 CPU、64 GiB，八组按 afterany 顺序运行。
normal 用于短 CPU 提交控制，不用于绕过 GPU debug 限制。

短测到完整训练的耗时校准复用上一轮四个模型实测：训练用完整训练日志中位耗时
与短测的比率上界，验证用完整验证日志耗时与短测的比率上界，再增加 10% 余量。
原始数值保存在 `timing_calibration.json`；每模型预算保存在其训练提交回执。

此处记录的是验收及自动提交流程已被调度器接受；截至本次查询，尚未声称
GPU 验收通过、八个正式训练作业已产生或新模型已有精度结果。

## 配置序列化修复与第二次提交（13:00 HKT）

首轮 MLoRE `4464680`、PLE `4464681` 在初始化时失败：MMEngine dump 重排
expert_groups 字典顺序，新模型错误地要求它与 decoder.groups 的顺序完全一致。
尚未开始训练，没有精度结果。修复为按名称集合核验；新增配置 dump/load 后的
真实小模型混合监督回归，相关 24 项测试通过。冻结新实现：
`bc744bd4907688628d009a7f196a4ca9dfa90bfa`。

仅取消了本轮仍 PENDING 的已知问题验收及控制作业；其他已有训练不受影响。
旧报告保留。新目录：
`/scr/u/xiaoz/HydroState/experiments/architecture_experts_20261118_v2`。

| 新 GPU 验收 | 作业号 |
|---|---:|
| 完整 MLoRE | 4464731 |
| PLE | 4464732 |
| 匹配 CGC | 4464733 |
| AdapterFusion E | 4464734 |
| CoSA Router | 4464735 |
| A／B／原 CGC 合并验收 | 4464736 |

新的自动训练提交作业为 **4464737**，依赖 4464736；脚本、环境、数据清单及
耗时校准均保存在新目录。最近查询 4464731 为 PENDING/Resources，其余等待依赖。
尚未声称第二轮 GPU 验收成功或正式训练已启动。

## 双分区调度

用户要求优先使用先可获得的资源。剩余 PENDING GPU 验收作业更新为
`Partition=c_foss_gpu,l40s`；单作业可由任一分区调度，不重复提交两份实验。
正式训练提交器新增 `--partitions c_foss_gpu,l40s`，仍使用 gpu QoS、单卡、4 CPU。
两分区分别提供 L40 与 L40S，实际运行分区/GPU 需随结果记录；吞吐不可直接混比。
调度顺序由 Slurm 优先级及可用资源决定，不承诺严格的先到先得。
模型、数据及配置仍冻结在 bc744bd release；更新的提交器单独绑定新 Git release。
MLoRE 4464731、PLE 4464732 已 COMPLETED/0:0；不重跑已完成验收。

提交器更新 commit：`383d733`。已取消仍排队且被 hold 的旧 CPU 控制任务
4464737，以 **4464833** 替代，仍依赖最后一个 GPU 验收 4464736。
新回执 `controller_dual_submission.json` 和调度变更记录
`dual_partition_update.json` 位于 v2 实验目录；原验收作业号与依赖不变。
仅提交器使用新 release，实际验收及训练模型源码仍为 bc744bd。

## 双分区 sbatch 拒绝修复

4464833 在提交 A 时失败，未产生训练 job。2026-11-18 复核：相同资源下
`sbatch --test-only --partition=c_foss_gpu,l40s` 返回 Invalid partition name，
两个单分区均通过。测试编号不算实际任务。HKU 的提交入口不接受该组合，
不能从此前 scontrol 更新成功推断 sbatch 也支持。
提交器改用单分区 `--hold` 提交，保存回执后 scontrol 更新双分区候选，成功才 release。
更新失败不会释放；已接受回执复用同一 job，不重复训练。保留原 A 拒绝记录后补交。
模型源码与科学配置仍使用 bc744bd，全部八个模型验收已通过，无需重做。
HPC skill 主文件及 Slurm 参考同步记录此站点限制和可用流程。

修复提交器冻结为 `4fd1797`，本地提交器相关 3 项测试、Ruff、skill 校验通过。
从登录节点执行一次补交，八组均接受且完成 hold→更新分区→release：

| 架构 | 六轮训练 job |
|---|---:|
| A | 4466178 |
| AdapterFusion | 4466179 |
| B | 4466180 |
| 原 CGC | 4466181 |
| 匹配 CGC | 4466182 |
| CoSA Router | 4466183 |
| 完整 MLoRE | 4466184 |
| PLE | 4466185 |

按表中顺序 afterany 串行执行；每个实验一个作业，候选分区 c_foss_gpu,l40s。
原 A 拒绝回执保存在 `A/submission_rejected_dual_partition.json`。
当前各 `submission.json` 保存实际提交命令、job ID、预算和 partition_dispatch=released。

## 独立排队（用户确认）

已对仍 PENDING 的 4466179–4466185 执行 `scontrol update JobId=... Dependency=0`，
核实八个训练作业 DEPENDENCY 均为 null。4466178 已运行，未中断或重启。
这覆盖上文的串行计划；模型之间没有科学或数据依赖，各自输出目录独立。
提交器同步移除模型之间的 afterany，仅保留提交前逐模型验收检查。
各任务保留单 GPU 和双分区候选，实际并行数量由可用资源和账号配额决定。

## 并行启动 OOM 诊断与隔离（2026-11-18）

原 CGC 4466181、CoSA Router 4466183、完整 MLoRE 4466184、PLE 4466185
在 SPG-3-2 启动约 48 秒后失败。错误中四个训练 PID 895383/895438/895427/895437
互相占用同一 GPU，单进程只用约 2.6–15 GiB 时卡已满，不能归因于 batch 太大。
CGC 同 batch=5 的独立验收峰值 reserved 30.86 GiB，GPU 总量 44.39 GiB。

只读诊断 4466263：SLURM_JOB_GPUS=3，CUDA_VISIBLE_DEVICES=0，但 nvidia-smi
可见两张 GPU（UUID 7d799632…、75fd8faf…），默认 CUDA 选择前者。
4466268/4466273 核实 /proc NVIDIA 设备含 minor 0–9，而 Slurm 节点配置为 8 GPU；
minor 3 对应 UUID 3fcfd8a2…，显式设该 UUID 后 torch.cuda.device_count()=0。
这证明分配标识与实际可访问设备不一致；不能安全地将 SLURM_JOB_GPUS 直接改成
CUDA_VISIBLE_DEVICES，也不能只强行指定一个看起来空闲的 UUID。
具体系统配置根因仍需集群管理员核实，本项目不改系统 GRES/cgroup 配置。

规避方案：四个失败模型保持原科学配置、batch 和有效 batch，仅在 l40s 独立补交；
四个正常运行模型不动。提交器增加 --variants，限定补交范围。
原失败任务日志及回执保留，禁止再次用原 job 的状态冒充成功。

修复提交器 release：`7fbbab3`。补交作业（l40s、单卡、无依赖）：
CGC **4466291**，CoSA Router **4466292**，完整 MLoRE **4466293**，PLE **4466294**。
各模型旧回执另存为 `submission_failed_gpu_<old_job>.json`；旧日志保留。
本次只修正运行位置及重试范围，模型仍为 bc744bd，无更改科学配置。
HPC skill Slurm 参考已记录节点隔离异常与诊断边界。

用户随后授权恢复双分区候选并排除异常节点。已直接更新四个仍 PENDING 的
4466291–4466294：`Partition=c_foss_gpu,l40s ExcNodeList=SPG-3-2`。
scontrol 已确认四者均无依赖、排除节点生效；沿用原 job ID，未重复提交。
这覆盖此前仅 l40s 的排队计划，不改模型、batch 或正在运行的任务。

用户先要求去除原 CGC，取消 4466291 已生效；随后决定保留。
已核实旧任务 CANCELLED 后按原配置补交 **4466456**，无依赖、单 GPU，
双分区候选且排除 SPG-3-2；旧回执保存为 `submission_cancelled_4466291.json`。
其余七组不变，本轮仍保留八个架构。

## 完整 MLoRE batch=2 重试

l40s 重试 4466293 在第 1 轮约第 40 批后 OOM，单进程占 44.23 GiB，
allocated 34.99 GiB、reserved unused 8.72 GiB；与此前多进程争抢不同，
本次为模型自身显存压力，短验收未覆盖后续高峰。
用户同意降低 batch 后，补交 **4467334**：micro-batch 5→2、accumulation 2→5，
有效 batch=10；训练/验证/开发集重载 batch 均为 2；LinearLR warmup 200→500
微批次，维持 100 次优化器更新。六轮、验证间隔、选模规则、模型与数据均不变。
MLoRE 的 BatchNorm 使用更小微批统计，梯度累积不能保证与 batch=5 完全等价。

独立输出：`MLoRE_complete/retry_batch2/`，配置、plan.json、submission.json 均保存；
原失败输出保留。无 checkpoint 可续接，从相同初始化方案重新开始。
复用已通过的 batch=2 实测速度、原 smoke 固定开销及既有耗时校准，增加 10%，
申请 04:50:00；这不是声称完整训练显存已获验证，仍需检查新运行。
双分区候选 c_foss_gpu,l40s，排除 SPG-3-2，无任务依赖，单 GPU、4 CPU。
