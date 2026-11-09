# CoSA 产品＋站点联合训练 — 2026-11-09

## 当前提交：SCR 数据使用 L40S

用户确认全部数据在 SCR，可改用 `l40s`。已核对产品、两个站点 cohort 的输入/标签及
FM 权重的实际解析路径都在 SCR；无须复制或重新下载。
旧控制器已停止，尚未启动的 `c_foss_gpu` 作业 **4414178 已取消**，旧运行目录保留。

新模型/启动器提交：`8f987a91679770d4fecde3523a3f4e53104ae919`。
新运行目录：`/scr/u/xiaoz/HydroState/experiments/cosa_joint_l40s_20261109`。
debug 作业 **4414428**：`l40s` / debug，单 `nvidia_l40s`，4 CPU，48 GiB 主存，25 分钟。
后续正式 B/C 也从运行记录读取相同分区、GPU 类型和 CPU/主存；时限仍按新 GPU 实测加 10%。
新的自动控制器已经启动；是否进入迭代以该目录的日志和 `controller_status.json` 为准。

本次只修改资源选择及校验，没有改变模型、采样、监督和 B/C 实验条件。
9 项提交控制相关测试、Ruff、Shell 语法及差异检查通过。

### 首次 L40S 调试发现的拼批问题及修复

4414428 在 SPG-5-4 启动，真实 SCR 混合批次预检通过，并完成三个 batch=16 GPU 更新。
随后随机批次由站点样本开头，DataLoader worker 的共享内存拼批遇到 Bool/UInt8 冲突，
作业以 FAILED 结束；控制器按预期停止，未提交正式训练。

原因是产品样本的空站点掩膜直接沿用了存储 uint8，实际站点标签则经公共读取器转为 bool。
已让占位标签复用相同 `as_tensors` 转换，不改变标签值、掩膜有效性或数据契约。
新增以两种来源交替开头、0/2 worker 的实际 Zarr 回归：修复前两种设置均复现，
修复后与 CoSA/联合监督共 21 项测试通过。后续重试采用独立目录
`/scr/u/xiaoz/HydroState/experiments/cosa_joint_l40s_20261109_r2`，保留失败日志。
batch 探测在 16–40 范围增加每 4 的候选，探测每档 4 步，最终 B/C 各 8 步吞吐测量、
4 步完整 smoke；显存余量和正式训练预算规则不变。记录每步显存以便诊断。

## 首次提交的代码和数据（已被上面的 L40S 提交替换）

模型源码冻结为 `84afe1fded3e4e4db97d9ff7647b710f16c60092`，已推送 GitHub 并同步 HPC。
发布目录为 `/scr/u/xiaoz/HydroState/releases/84afe1fded3e4e4db97d9ff7647b710f16c60092`。
运行根目录：`/scr/u/xiaoz/HydroState/experiments/cosa_joint_20261109_84afe1f`。

- 产品：`/scr/u/xiaoz/HydroState/samples/retrieval_eight_20000_06aba36`。
- 站点：`/scr/u/xiaoz/HydroState/samples/site_joint_v3_20261108`，发布标记 passed。
- 原始 FM：SCR 下 `pretrained/OlmoEarth-v1_2-Base`；不继承历史任务训练权重。
- 数据、v3 划分及训练统计只读；保留两个站点生产批次各自的指纹。

B 为 `configs/hydrostate_joint_independent.py`，C 为 `configs/hydrostate_joint.py`，
只切换 Router。站点 SM/ET 标准化 MSE 系数均为 2；选模站点占 2/3、产品占 1/3。
产品/站点抽样权重各半，站点任务→空间组→日期，频率组仅用于报告。
新增站点空间组宏平均和共同年份指标；其结果不改变固定选模公式。

## 本地检查

38 项 CoSA、联合监督、入口和 checkpoint 测试通过。另有 8 项新旧提交控制测试通过，
覆盖通过条件、显存余量、训练配方一致性、去重、续跑和仅最终一次完整测试。
Ruff、差异空白检查和 Shell 语法检查通过。

## 首次排队记录

debug 作业 **4414178** 已接受提交：`c_foss_gpu`、debug、单 L40、4 CPU、48 GiB 主存，
25 分钟上限。提交时排队等待资源，不能称为 GPU 验证通过或正式训练已经启动。

debug 先核验真实 SCR 批次，再递增探测 CoSA C 的 batch；要求峰值 allocated/reserved
均保留 10% 显存余量。用选中 batch 对 B/C 各做 20 步读取＋训练测量和代表性验证，
随后各自检查混合监督训练、任务头梯度/更新、保存与重载和抽样测试。测试权重不用于正式初始化。

显式启动的 `tools/experiments/launch_joint.py --watch` 在 IO 节点等待 debug，只有作业成功、
两份验收报告通过、相同 batch、固定站点权重和显存余量均满足时才提交正式 B/C。
正式各 50 联合 epochs、单 L40/4 CPU，保持相同有效 batch；每个 epoch 为 43,266 次抽样。
walltime 从代表性实测训练/验证吞吐和短测固定开销估算，加 10%；超过七天则在 checkpoint
边界续跑分段。这里只是实测子集外推，完整数据的 I/O 可能不同，不能宣称已测完一个正式 epoch。
最后一段训练成功后，对最佳验证 checkpoint 做一次完整测试。

控制器版本与模型源码分别绑定，不修改运行中的冻结模型目录。状态来源：

- `submission.json`：配置、源码、环境、作业 ID、资源和测量。
- `preflight.json`：划分/统计指纹、实际批次及采样质量。
- `probe_c_*.json`、`benchmark_b.json`、`benchmark_c.json`：显存与吞吐。
- `smoke_b/result.json`、`smoke_c/result.json`：数值和 checkpoint 验收。
- `controller_status.json` / `controller_failure.json`：后台控制器实时状态或停止原因。
- `formal_intent_*` / `formal_receipt_*`：提交去重及 Slurm 回执。

截至此记录建立时，尚无新 CoSA 正式验证精度。历史 A 仅作为产品-only 参考，
与本轮比较同时包含监督来源变化；不能将差异全部归于 CoSA 架构。
