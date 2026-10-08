# 可配置的 HydroState 架构

默认模型仍为 `HydroStateModel`。`StagedHydroStateModel` 实现三阶段任务分支、
交互前辅助头和一次同步门控交互。两者使用同一个 OlmoEarth 编码器、数据合同、
30 m 输出、产品/站点监督支持、训练集标准化统计和最终 NLL；评估和预测 API 相同。
新增结构不依赖新的第三方包。这里的公式描述具体实现，尚不代表性能优于基线。

## 配置入口

| 配置 | 结构 |
| --- | --- |
| `configs/hydrostate_l40s.py` | 原共享解码器 + 三个最终头 |
| `configs/hydrostate_staged.py` | 三阶段 CBAM + bottleneck Adapter + 辅助头 + Sigmoid 门控 |
| `configs/hydrostate_staged_light.py` | 同一完整 Sigmoid 结构，通道缩窄为 256→128→64 |
| `configs/hydrostate_staged_independent.py` | 同上，关闭跨任务交互 |
| `configs/hydrostate_staged_softmax.py` | 同上，改为 Softmax 门控 |

所有新增配置继承 L40S 的数据和优化设置。每个预设有独立默认 work_dir；若设置了
`HYDROSTATE_WORK_DIR`，需要为每次实验分别设置目录。编码器使用原 OlmoEarth 预训练
初始化；新的分支、门控、辅助头和最终头按 PyTorch 默认方式初始化。

```bash
python tools/train.py configs/hydrostate_staged.py
python tools/train.py configs/hydrostate_staged_light.py
python tools/train.py configs/hydrostate_staged_independent.py
python tools/train.py configs/hydrostate_staged_softmax.py
```

也可在派生配置或现有 CLI 中组合数学选项：

```bash
python tools/train.py configs/hydrostate_staged.py --cfg-options \
  model.decoder.attention.kind=se \
  model.decoder.adapter.kind=depthwise \
  model.decoder.fusion=concat \
  model.interaction.mode=softmax \
  model.auxiliary_weight=0.0 \
  work_dir=/tmp/hydrostate_staged_ablation
```

| 配置字段 | 可选值 | 默认 |
| --- | --- | --- |
| `model.decoder.attention.kind` | `none`, `se`, `cbam` | `cbam` |
| `model.decoder.attention.reduction` | 正整数，通道瓶颈缩减倍数 | 16 |
| `model.decoder.adapter.kind` | `none`, `bottleneck`, `depthwise` | `bottleneck` |
| `model.decoder.adapter.reduction` | 正整数 | 4 |
| `model.decoder.fusion` | `add`, `concat` | `add` |
| `model.decoder.merge_ops` | `True`, `False`；合并独立任务算子 | `True` |
| `model.interaction.mode` | `none`, `sigmoid`, `softmax` | `sigmoid` |
| `model.interaction.merge_ops` | `True`, `False`；合并消息投影及加权求和 | `True` |
| `model.interaction.residual_scale` | 有限非负数 | 0.1 |
| `model.auxiliary_weight` | 非负标量，或三任务权重字典 | 各任务 0.2 |

权重字典必须包含 `water`, `soil_moisture`, `et` 三个键。权重为零的辅助头不创建。
关闭注意力、Adapter 和交互后仍存在逐级分支投影，**不等价于旧基线**；严格基线应使用
`hydrostate_l40s.py`。预设只切换架构，不自动选检查点或混用不同数据集的统计。

## 逐级任务特征

共享主干保持原结构：

```text
OlmoEarth 输出 768×30×30
  → S1: 512×30×30
  → S2: 256×60×60
  → S3: 128×120×120
```

对每个任务 t 独立建立三个阶段。第一阶段 `X_t^1 = S1`。
后续阶段先投影上一阶段任务特征，再双线性上采样到当前共享特征大小：

`U_t^k = Resize(Conv1x1_t,k(T_t^(k-1)))`。

- `add`: `X_t^k = S^k + U_t^k`。
- `concat`: `X_t^k = Conv1x1_t,k([S^k, U_t^k])`。

然后 `T_t^k = Adapter_t,k(Attention_t,k(X_t^k))`。
三个任务、三个阶段各有独立参数；共享主干不接收其他分支的前向输入，但会接收所有
任务损失的梯度。`F_t = AvgPool3x3,stride3(T_t^3)`，形状为 `128×40×40`。

注意力使用通道/空间乘性门控，避免在 120×120 网格创建全局 token 两两注意力矩阵：

- `se`: `X * sigmoid(MLP(AvgPool_spatial(X)))`。
- `cbam`: 通道门控为 `sigmoid(MLP(AvgPool(X)) + MLP(MaxPool(X)))`，
  两种池化共用 MLP；随后将通道均值和最大值拼接，经 7×7 卷积和 Sigmoid 得到空间门控。
  此处 MLP 采用 GELU，是本项目的 CBAM 变体。
- `none`: 恒等映射。

Adapter 为残差结构：

- `bottleneck`: `X + Conv1x1_up(GELU(Conv1x1_down(X)))`。
- `depthwise`: 在瓶颈中增加 `DepthwiseConv3x3 + GELU`，其余相同。
- `none`: 恒等映射。

## 轻量化通道配置

`hydrostate_staged_light.py` 继承完整 Sigmoid 配置，仅调整通道及独立输出目录：

```text
OlmoEarth: 768×30×30
共享及任务阶段: 256×30×30 → 128×60×60 → 64×120×120
每任务平均池化: 64×40×40
辅助头 / 一次 Sigmoid 交互 / 最终头: 输入均为 64 通道
输出: WF、SM、ET 各 1×40×40，仍对应 30 m 网格
```

CBAM、Adapter、辅助监督及输出激活保持原设置；保留算子合并、CPU 时相元数据和
固定形状损失优化。数据、标准化统计、batch size、优化器及编码器配置从父配置继承。
共享主干单样本前向卷积 MACs 从 18.0486 G 降为 5.3084 G，减少约 70.6%；此统计不含
编码器及反向传播，也不代表整网速度提高同等比例。减少通道改变了模型容量，需要
独立评估精度；它不是数学等价优化。宽版解码器检查点尺寸不兼容，不能直接续训到
轻量版；默认依旧从 OlmoEarth 预训练编码器初始化新模型。

使用现有 HPC bash 启动链时，将 `HYDROSTATE_CONFIG` 设为此配置；既有宽版配置及
冻结训练快照保持原样。交互消融仍可用 `--cfg-options model.interaction.mode=none`
或 `model.interaction.mode=softmax` 选择。

## 一次同步门控交互

在 40×40 网格，三任务之间共有六条有向边。对每条 `s → t`：

`V_st = Conv1x1_st(F_s)`，`a_st = Conv1x1_gate,st([F_t, F_s])`。

V 有 128 个通道；a 是每个位置一个标量，在特征通道间广播。

- `sigmoid`: `F'_t = F_t + alpha/2 * sum_(s!=t) sigmoid(a_st) * V_st`。
- `softmax`: 对两条输入边的 a 做 Softmax，
  `F'_t = F_t + alpha * sum_(s!=t) softmax_sources(a)_st * V_st`。
- `none`: `F'_t = F_t`。

`alpha` 为配置中的固定 `residual_scale`。所有消息都读取交互前的 F，构造独立输出，
没有顺序更新或第二轮交互。这里传递的是特征，不是物理量；无显式水量守恒约束。

## 辅助头、最终头与损失

辅助头读取交互前的 F，最终头读取交互后的 F'。两类头拥有独立参数，结构均为
`Conv3x3 → GELU → Conv1x1 → activation`；WF、SM 为 Sigmoid，ET 为 Softplus。
只有 `mode='loss'` 执行辅助头；`mode='tensor'`、`predict` 和验证仅返回三个最终输出。

`L = sum_t NLL_t(final) + sum_t lambda_t * MSE_t(aux)`。

辅助 MSE 使用冻结训练标准差：`mean(((prediction-target)/std_t)^2)`。
它不学习额外方差，也不直接更新最终 NLL 的 log variance。
辅助与最终输出使用完全相同的有效掩膜和监督支持：WF 30 m 像元、SM 观测窗口均值、
ET 完整原生像元面积加权均值，以及可选站点足迹。缺测标签在运算前被排除；
某任务全局无有效标签时该任务损失为零。多卡按全局有效观测数归一化并补偿 DDP 梯度平均。

新增解码分支、交互模块、辅助头使用完整基础学习率；编码器继续使用原分层衰减。
辅助头会随训练 checkpoint 保存，便于恢复训练，但推理不会执行它们。
原架构的 state_dict 键未更名，可继续严格加载原架构 checkpoint。新架构新增参数，
不能把旧 checkpoint 当作完整新模型恢复；跨架构热启动须显式检查 missing/unexpected keys。
用训练时的同一配置评估新模型，不要切换架构后直接 resume。

## 数学等价的算子合并

`merge_ops=True` 不使用 `torch.compile`，不改变通道宽度、空间网格、任务参数或损失。
CBAM 的平均/最大池化统计在 batch 维合并后通过同一个 MLP；同级任务投影和 Adapter
使用分组卷积，并合并双线性上采样。任务注意力仍分别计算，空间统计不跨任务混合。
交互中每个来源的两条消息投影合并为双输出卷积，仍保留六套独立权重；加权和避免
堆叠两个完整消息张量。Softmax 与 Sigmoid 的归一化规则均保持原样。

原 Parameter 对象、注册顺序及 state_dict 键不变；拼接权重通过 autograd 回传至各自
参数，因此可严格加载合并前的 staged 模型权重和对应优化器状态。浮点舍入和卷积
后端选择可能导致小的数值差异，不保证逐位一致。局部 Conv2d 模块的 forward hook
不会捕获经 functional 分组卷积执行的调用；统计算子数请用 profiler。

同时设置 `model.decoder.merge_ops=False` 和 `model.interaction.merge_ops=False` 可恢复
原执行方式。若显式指定了 `model.decoder.attention.merge_ops`，该值优先于解码器开关。
算子合并并不降低理论卷积 MACs，也不保证在所有设备上加速；需要以实际吞吐为准。
已提交 HPC 作业使用冻结代码快照，本地修改不会自动更新正在运行的作业。

## CPU 时相元数据与固定形状损失

数据预处理器在设备搬运前，根据 CPU 上的 `sensor_valid` 和 `observation_time`
准备各样本/模态的整数时相索引与 UTC 时间戳。所有索引打包为一个张量后搬到 GPU，
编码器使用 `index_select`，不再逐模态读取 CUDA 布尔值或转换 CUDA 张量为 Python
列表。原始时间控制张量保留在 CPU；上游 rslearn 的真实时间对齐算法保持不变。
直接调用编码器而绕过预处理器时，兼容路径最多整体复制两个时间控制张量到 CPU。
分组时相实验在拆分批次后重建索引，避免复用原批次偏移。

每批只准备一次 WF、SM、ET 与站点监督的有效范围、覆盖完整性、池化分母和有效计数，
最终头与辅助头分别投影预测并复用这些信息。NLL/MSE 使用固定形状掩膜归约，先清除
无效位置的预测及标签，再做算术运算；全缺测任务保持可微的零损失和零梯度。
NLL 保留原标准化顺序、Gaussian 常数项、方差下限及梯度规则。

多卡时把最终/辅助头的局部损失和三任务计数合并到一次 all-reduce，两个头引用同一
全局任务分母，并继续补偿 DDP 梯度平均。合并的只是脱离梯度图的统计信息：单独反传
辅助损失仍不会更新最终头或最终 NLL 方差。上述变化不使用 `torch.compile`。

## 验证与训练范围

测试覆盖三阶段输出、数学选项、六向同步交互公式、跨任务梯度、辅助头只在 loss 执行、
多尺度/站点支持、缺测标签、优化器分组、checkpoint 和配置继承。
离线测试使用替代编码器隔离新增结构，不能替代真实 OlmoEarth 与数据的 GPU 训练验证。
L40S batch=16 来自旧架构；新增结构的显存和吞吐需要实测。若减小 microbatch，应同步
调整梯度累积及按 iteration 计数的 warmup，保持对照实验的有效批次和更新预算一致。
独立的 `sm_causal` 启动器仍面向它原有模型，未自动合并到本架构。

2026-10-08 本地真实数据检查：使用本地 OlmoEarth v1.2 Base 权重与多尺度 pilot 中
同时具备三任务标签的一个样本，RTX 4060 / BF16 / batch=1 完成一次 AdamW 更新和
验证前向。编码器末层、共享主干、三个任务分支、三个辅助头、三个最终头及交互模块
所检查的参数均发生更新；三个输出均为 `1×40×40` 且数值有限。模型含 124,234,770
个参数，峰值 PyTorch allocated 显存约 1775 MiB。本次仅验证执行，使用既有训练快照
的固定标准化统计，没有保存训练 checkpoint，也不能据此推断精度或 L40S batch=16
显存。报告位于 `outputs/staged_architecture_smoke/gpu_smoke.json`。

## 已有合并样本的 HPC 提交

`scripts/slurm/24_submit_staged_training.sh` 复用现有的
`22_train_merged_l40s.sh → 12_train_l40s.sh → 11_train_hpc.sh` 启动链。
它直接使用已准备好的 `samples_merged_v1/manifests/merged_v1_n20000`，不重做数据准备，
先提交 `hydrostate_staged_smoke.py`（8 次更新、全量验证、限时 2 小时），再提交
依赖其成功完成的 `hydrostate_staged.py`（50 epochs、限时 96 小时）。均为单张 L40S、
batch=16、BF16；正式训练从预训练编码器重新初始化，不从 smoke checkpoint 续训。
smoke 同时核验新增分支、辅助头和交互模块的参数更新。

建议先冻结源代码并验证与基线一致的数据指纹和训练设置，再运行：

```bash
HYDROSTATE_CODE_ROOT=/path/to/frozen/code \
  bash /path/to/frozen/code/scripts/slurm/24_submit_staged_training.sh
```

提交记录写入该代码目录的 `staged_submission.txt`，重复执行会拒绝重复提交。
若部分提交失败，先读取记录及 Slurm 状态，再处理缺失步骤。
2026-10-08 已提交 smoke `4243167`、正式训练 `4243168`；代码快照在 HPC 的
`/home/xiaoz/HydroState/submissions/staged_sigmoid_20261008_173337`，其中
`preflight.json` 保存数据哈希、基线配置一致性检查和架构参数。

随后按用户要求，正式任务 `4243168` 已移除 smoke 依赖并解除挂起，排队中的
smoke `4243167` 已取消。正式设置保持 batch=16、累积=1、BF16、50 epochs、96 小时；
候选分区扩展到 `l40s,c_foss_gpu,condo_gpu`，要求 `GPU_MEM:48GB`，不使用 interactive。
手动桌面测试完成了 128 个微批次并观察到参数更新，但在保存 checkpoint 时被终止，
未完成保存/全量验证；此次直接开始正式训练是用户明确选择，并非完整 smoke 已通过。
最新决策记录见本地 `outputs/staged_submission_20261008_173337/jobs.json` 和远端快照内
`training_release.json`。上述提交脚本仍默认采用 smoke 依赖，本次是已有 Slurm 作业的调整。
