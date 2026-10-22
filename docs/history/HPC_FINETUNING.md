> 历史记录：不作为当前执行依据。最新入口见 [当前文档](../README.md)。

# HPC 产品数据微调：训练集标准化与不确定性 NLL

当前使用 `/home/xiaoz/miniconda/envs/hydrostate` 环境，支持单张或多张 L40S。
启动脚本按作业实际可见的 GPU 数量自动选择单卡训练或 DDP；当前默认申请一张。
L40S 节点不能访问 `/fossfs`，数据、预训练权重和训练输出都使用 `/scr/u/xiaoz`。
原始数据保留在 `/fossfs/xiaozhen/HydroState/samples_v1`。
上传新数据的流程见 [HPC_SAMPLE_UPLOAD.md](HPC_SAMPLE_UPLOAD.md)。

## 数据与固定统计量

2026-09-22 的快照有 1024 个样本：train 747、validation 121、test 156。
清单为 `/scr/u/xiaoz/HydroState/samples_v1/manifests/gee_20260922_n1024/samples.parquet`。
同目录的 `target_stats.json` 只统计训练集，采用总体标准差（`ddof=0`），
每个训练样本读取一次，关闭随机增强，不使用验证集、测试集或重复采样频次。

| 任务 | 有效监督数量 | 均值 μ | 标准差 σ | 单位 |
| --- | ---: | ---: | ---: | --- |
| water | 1,157,072 | 0.2576189443 | 0.4296761804 | 水体比例 |
| soil_moisture | 167 | 0.2175518853 | 0.1085070754 | m³/m³ |
| et | 2,735 | 1.0797257770 | 1.2453036163 | mm/day |

统计使用与损失一致的有效标签和输入覆盖掩码：water 是完整的 30 m 聚合像元；
SMAP 是有效窗口中心标签；ET 是完整覆盖的 PML 原生像元。当前没有有效站点标签。
SMAP 仍是窗口预测均值对应中心标签的弱监督；学习任务方差不能修复空间支持偏差。
本快照未覆盖历史水体频率为 100% 的第 6 层，适合阶段性微调验证。

统计脚本以 float64 合并矩，拒绝有效标签中的非有限值、过少标签和近零标准差。
JSON 包含统计口径版本、清单 SHA256、训练记录 fingerprint 和有效数量。
训练入口校验训练记录 fingerprint，并将完整统计内容嵌入运行配置；
相同统计内容同时传给模型及验证/测试评估器。这依赖 Zarr 快照不可变的约定，
fingerprint 验证清单记录，不逐次重新哈希 Zarr 内容。

对新的、独立命名的数据快照，先运行一次：

```bash
cd /home/xiaoz/HydroState
export HYDROSTATE_DATA_ROOT=/scr/u/xiaoz/HydroState/samples_v1
export HYDROSTATE_MANIFEST="$HYDROSTATE_DATA_ROOT/manifests/NEW_SNAPSHOT/samples.parquet"
sbatch scripts/legacy/slurm/04_target_statistics.sh
```

已有 `target_stats.json` 时脚本拒绝覆盖。已有实验始终复用其固定统计文件；
扩充或改变训练记录时使用新快照、新统计文件及新实验目录。

## 当前目标函数

sigmoid 水体/土壤水分头及 softplus ET 头仍输出物理量。
OlmoEarth 输入标准化保留在 Zarr 生成阶段；这里标准化的是监督变量：

```text
z_pred = (prediction - μ_train) / σ_train
z_true = (target     - μ_train) / σ_train
v_t = exp(max(s_t, log(1e-6)))        # 每个任务一个可学习 s_t，初始为 0
L_t = mean_valid[0.5 * ((z_pred - z_true)^2 / v_t + log(v_t) + log(2π))]
L = L_water + L_soil_moisture + L_et
```

这是每个任务一个同方差高斯 NLL，参考
[Kendall、Gal、Cipolla 的多任务不确定性加权方法](https://arxiv.org/abs/1705.07115)，
误差项通过 [PyTorch GaussianNLLLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.GaussianNLLLoss.html)
计算。`1/v_t` 控制该任务标准化平方误差的权重，`log(v_t)` 惩罚无限放大方差。
方差描述任务整体的标准化残差水平，不是逐像元不确定性，也不是已校准的置信区间。
它同时反映噪声、产品偏差和模型当前拟合能力。

预测和标签都减去同一均值，因此均值在残差中相消；均值仍被固定保存，
标准差决定残差的尺度。旧任务权重 `1/0.25/1`、旧物理缩放 `1/0.1/3`
以及站点额外乘数已从当前模型配置移除。未来有效站点观测采用相同任务统计量和方差，
与产品观测合并后计算该任务均值，不再额外乘 3；站点 footprint 必须完整观测。

掩码先于 NLL 运算应用。一个任务在全局批次没有有效观测时，误差项和方差项都为零；
两卡有效数量不同时，按全局有效数量归一化，并补偿 DDP 的梯度平均。
梯度累积平均各 microbatch 的目标值。每个任务先取均值再相加，避免水体像元数量
直接压过 SMAP/ET。三个 `log_variances` 使用基础学习率且不施加 weight decay。

日志记录 `loss_*`、`valid_count_*`、`variance_*` 和 `precision_*`。
NLL 可以为负，不能按损失正负判断是否存在标签。评估报告物理单位的 MAE/RMSE/bias/R²；
`hydro_score = -mean(RMSE_t / σ_train,t)`，不随可学习方差改变模型选择标准。

## 通过 SLURM 验证

CPU 数值、掩码、数据隔离及 DDP 梯度测试：

```bash
sbatch scripts/legacy/slurm/05_test_training.sh
```

两张 L40S 的短训练使用现有 `hydrostate` 环境，默认路径均已指向 scratch：

```bash
sbatch --gres=gpu:nvidia_l40s:2 --qos=debug --time=00:30:00 --job-name=hs_nll_smoke \
  --export=ALL,HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_l40s_smoke.py \
  scripts/legacy/slurm/12_train_l40s.sh
```

当前短训练配置跟随正式配置：每卡 batch size 16、梯度累积 1、8 次迭代，
即 8 次优化器更新，
随后评估全部 121 个验证样本并保存 checkpoint。检查 encoder、decoder、三个头和
三个方差参数均更新且两卡一致，统计量保持不变，并从 checkpoint 恢复 NLL 状态。
结果写入 `/scr/u/xiaoz/HydroState/work_dirs/nll_JOBID/nll_verification_rank*.json`。
日志位于 `/home/xiaoz/HydroState/logs/`；此检查不读取 test 集。

2026-09-22 已完成：统计作业 `4098289`，40 项测试作业 `4098315`，
两卡短训练及完整 validation 作业 `4098316`，均以 exit code 0 完成。
该已完成作业使用旧配置：每卡 batch size 2、梯度累积 8、64 次 microbatch。
后者报告为 `/scr/u/xiaoz/HydroState/work_dirs/nll_4098316/nll_verification_rank0.json`。
这验证了执行与数值正确性，8 次更新不足以证明学习到最终合适的权重或检索精度。
新的每卡 batch size 16 按用户要求采用估计值，尚未进行显存或吞吐量实测。
正式 50 epoch 训练尚未启动。

## 正式训练与恢复

正式配置是 `configs/experiments/legacy/hydrostate_l40s.py`：每卡 batch size 16，梯度累积 1；
有效 batch 为 `16 × GPU 数量`，单卡为 16，双卡为 32。
全量更新 encoder、共享 decoder 和三个任务头，并学习三个方差参数。
AdamW 基础 LR 1e-4、weight decay 0.05，encoder 逐层 LR 衰减系数 0.65；
BF16、梯度裁剪 1、50 epoch、每 epoch 验证，以 `hydro_score` 保存最佳 checkpoint。
线性预热为 125 次训练迭代，对应原配置 1000 次 microbatch / 累积 8，
保持约 125 次参数更新的预热长度；余弦退火仍按 50 epoch 配置。
验证和测试每卡 batch size 保持 2。这些是起始超参数，尚未经过完整训练调优。

```bash
# 正式实验：使用独立目录和正式配置。
export HYDROSTATE_CONFIG=configs/experiments/legacy/hydrostate_l40s.py
export HYDROSTATE_WORK_DIR=/scr/u/xiaoz/HydroState/work_dirs/gee_20260922_nll_v1
# 当前默认申请 1 张 GPU，自动使用单进程 --launcher none。
sbatch scripts/legacy/slurm/12_train_l40s.sh

# 或申请 2 张 GPU，自动使用 torchrun 和两个 DDP 进程；两条命令择一执行。
sbatch --gres=gpu:nvidia_l40s:2 scripts/legacy/slurm/12_train_l40s.sh

# 单卡实验中断后的恢复，保留同一份配置、统计量和 work_dir：
sbatch scripts/legacy/slurm/12_train_l40s.sh resume=True
# 双卡实验恢复时同样加上 --gres=gpu:nvidia_l40s:2。
```

`11_train_hpc.sh` 使用 `torch.cuda.device_count()` 读取作业内的可见 GPU，
遵循 CUDA/SLURM 的设备隔离。无需同时手动修改进程数，启动日志会打印识别数量和模式。
`HYDROSTATE_GPUS_PER_NODE` 默认自动检测；若显式设为整数，则校验其与可见 GPU 数量一致。
旧终端若保留了该变量，可执行 `unset HYDROSTATE_GPUS_PER_NODE` 恢复自动检测。
没有可见 CUDA GPU、显式数量不匹配或不支持所选 BF16 时，仍会在启动前报错。
自动选择启动方式不会自动改变每卡 batch size、梯度累积或学习率。

### 每轮固定样本可视化

L40S 正式配置默认启用 `HydroVisualizationHook`，每个 epoch 验证后保存 4 个固定验证样本。
优先选择 WF、SM、ET 三个任务都有有效监督的样本，再按 `sample_id` 排序；
不足时用有较少有效任务的样本补齐。不可变验证集下，跨 epoch、恢复训练及不同 GPU 数量
均选择同一组样本。无效区域显示为灰色，每个任务各列及不同 epoch 使用相同色标。

每个样本生成一张九宫格 PNG，行依次为 WF、SM、ET，列依次为：

| 任务 | 30 米预测 | 监督尺度预测 | 真实值 |
| --- | --- | --- | --- |
| WF | 30 米输出 | 有效标签掩码内的 30 米预测 | 从输入网格聚合到 30 米的标签 |
| SM | 30 米输出 | 有效窗口预测均值，单个标量 | SMAP 中心观测值，单个标量 |
| ET | 30 米输出 | 完整覆盖的原生像元面积加权预测 | 同一原生像元的 PML 观测值 |

监督尺度预测直接使用模型计算损失/评估时的聚合结果。SM 标量面板不是空间真值图；
ET 原生像元预测与真值按相同 footprint 投影到显示网格，边界按存储的面积权重混合，
不代表新增的 30 米观测。WF、SM 色标范围为 0–1，ET 默认为 0–10 mm/day；超范围颜色饱和。

```bash
# 每轮展示 8 个固定验证样本（设为 0 关闭）。
HYDROSTATE_VIS_NUM_SAMPLES=8 sbatch scripts/legacy/slurm/12_train_l40s.sh

# 也可直接覆盖 hook 参数；这里展示 6 个样本，并扩大 ET 色标范围。
sbatch scripts/legacy/slurm/12_train_l40s.sh custom_hooks.0.num_samples=6 custom_hooks.0.et_vmax=15
```

结果位于 `$HYDROSTATE_WORK_DIR/visualizations/epoch_001/`、`epoch_002/` 等目录，
每个目录附带 `samples.json`，记录样本 ID、时间及有效监督数量。
请求数量超过验证集大小时仅输出现有样本并记录警告。
Hook 复用验证推理，多卡汇总后只由 rank 0 写图，不增加模型前向计算。
短训练配置也保留该 hook，输出到 `visualizations/iter_000008/`。
若修改 `val_interval`，只在执行验证的轮次输出；`custom_hooks.0.interval` 可进一步控制输出间隔。

训练结束或中途需要用最新检查点重新导图时，可单独提交验证作业：

```bash
# 最后一个参数是图片数量，省略时默认 4。
sbatch scripts/legacy/slurm/19_export_checkpoint_images.sh \
  /scr/u/xiaoz/HydroState/work_dirs/nll_JOBID 8
```

脚本读取训练目录中的 `last_checkpoint` 和保存的 `.py` 配置，使用原验证集及固定统计量，
只执行验证，不继续训练。输出目录为
`训练目录/checkpoint_visualizations/检查点名称/epoch_000/`；
`epoch_000` 是独立验证的编号，实际权重来自所选检查点。
训练目录中有多个 `.py` 配置时，用 `HYDROSTATE_EXPORT_CONFIG` 指定；
也可用 `HYDROSTATE_EXPORT_CHECKPOINT` 指定某个 checkpoint，
`HYDROSTATE_EXPORT_DIR` 指定输出目录。三者的相对路径均相对于训练目录。
完整参数说明：`bash scripts/legacy/slurm/19_export_checkpoint_images.sh --help`。

归一化均值、标准差、训练 fingerprint 与三个可学习方差均在 checkpoint 中；
完整统计报告同时位于运行配置中。加载时若配置与 checkpoint 的归一化不一致会报错。
旧 Huber 训练 checkpoint 不具备这份状态，不能直接作为本方案的 resume checkpoint。
测试/预测应使用实验保存的完整运行配置及最佳 checkpoint，复用训练统计量，
不能从验证/测试数据重新估计统计量。正式评估使用原有 test 划分。

当前环境为 Python 3.12.14、torch 2.7.1+cu128、torchvision 0.22.1+cu128，
已执行真实模型检查。本次工作没有升级环境；`pyproject.toml` 中的更新依赖范围
尚未与该现有环境锁定配置对齐，勿直接重装依赖来复现本次运行。
