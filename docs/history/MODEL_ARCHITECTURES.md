> 历史记录：不作为当前执行依据。最新入口见 [当前文档](../README.md)。

# 可配置的 HydroState 架构

新增 [Shared / Private LoRA 架构](SHARED_PRIVATE_MODEL.md) 通过
`configs/experiments/legacy/hydrostate_shared_private.py` 选择，注册类为 `SharedPrivateHydroStateModel`。
它冻结 OlmoEarth 基础权重，支持分阶段训练与新增变量适配；下文旧配置保持原行为。

规划中的[小型 CNN 独立专家权重空间实验](CNN_EXPERT_WEIGHT_SPACE_EXPERIMENT_PLAN.md)
比较仅 OlmoEarth 与多 FM 来源的 CNN 初始化；该方案尚未实现，不属于下列可运行架构。

默认模型仍为 `HydroStateModel`。`StagedHydroStateModel` 实现三阶段任务分支、
交互前辅助头和一次同步门控交互。两者使用同一个 OlmoEarth 编码器、数据合同、
30 m 输出、产品/站点监督支持、训练集标准化统计和最终 NLL；评估和预测 API 相同。
新增结构不依赖新的第三方包。这里的公式描述具体实现，尚不代表性能优于基线。

## 配置入口

| 配置 | 结构 |
| --- | --- |
| `configs/experiments/legacy/hydrostate_l40s.py` | 原共享解码器 + 三个最终头 |
| `configs/experiments/legacy/hydrostate_staged.py` | 三阶段 CBAM + bottleneck Adapter + 辅助头 + Sigmoid 门控 |
| `configs/experiments/legacy/hydrostate_staged_light.py` | 同一完整 Sigmoid 结构，通道缩窄为 256→128→64 |
| `configs/experiments/legacy/hydrostate_staged_light_land_sm.py` | 轻量版 + WF 约束 SM 适用域和陆地面积加权监督 |
| `configs/experiments/legacy/hydrostate_staged_independent.py` | 同上，关闭跨任务交互 |
| `configs/experiments/legacy/hydrostate_staged_softmax.py` | 同上，改为 Softmax 门控 |
| `configs/experiments/legacy/hydrostate_multilevel.py` | 编码器第 4/8/12 层融合 + 轻量共享解码器 + 30 m Adapter + 可学习强度 Sigmoid 门控 + 陆地条件 SM |
| `configs/experiments/legacy/hydrostate_multilevel_independent.py` | 多层融合版，保留任务 Adapter，关闭交互 |
| `configs/experiments/legacy/hydrostate_multilevel_shared.py` | 多层融合版，关闭 Adapter 与交互，直接使用共享特征预测 |

所有新增配置继承 L40S 的数据和优化设置。每个预设有独立默认 work_dir；若设置了
`HYDROSTATE_WORK_DIR`，需要为每次实验分别设置目录。编码器使用原 OlmoEarth 预训练
初始化；新的分支、门控、辅助头和最终头按 PyTorch 默认方式初始化。
例外是新增多层融合版的 Adapter：最后一层零初始化，初始为恒等映射；
其交互残差强度为各任务独立的可训练标量，初值 0.1，且没有辅助头。

## 多层融合版（新增）

```text
S1/S2/Landsat 多时相、时间戳、有效性掩膜
  → 可训练 Patch Embedding + 官方位置/时间/模态编码
  → OlmoEarth v1.2 Base 的第 4 / 8 / 12 个 Transformer block
  → 每层依有效掩膜聚合时相和 bandset，再对该位置可用模态取平均
  → 三个 768×30×30 特征（浅→深）
  → 深层投影 + 卷积：256×30×30
  → 降维上采样 + 中层投影跳接 + 卷积：128×60×60
  → 降维上采样 + 浅层投影跳接 + 卷积：64×120×120
  → AvgPool(3, stride=3)：64×40×40
  → 三个独立残差 Adapter：64→16→16→64
  → 一次同步 Sigmoid 门控交互，三个独立可学习残差强度
  → 三个预测头：Conv3×3(64→32) → GELU → Conv1×1(32→1)
  → Sigmoid WF / Sigmoid SM_land / Softplus ET
```

`MultiLevelOlmoEarthEncoder` 在一次编码中读取中间结果，不重复运行三个编码器。
`out_indices=[3,7,11]` 为从零开始的层索引，必须递增并包含最后一个 block；
当前融合解码器要求三个输出层。各层使用同一个预训练最终 LayerNorm 归一化。
该实现复用官方 patch embedding、时空编码、Transformer 和 token 恢复方法，
支持 RoPE 与梯度检查点；使用 PyTorch SDPA，不支持外部 packed flash-attn 路径。
没有增加 torch.compile。

缺失时相先从 attention 序列中移除，batch 补齐位置在训练和验证时均作为无效 key
屏蔽。池化时先计算有效时相/bandset 均值，再计算实际可用模态均值；
缺失整种模态不会作为零值拉低特征。旧单层编码器的池化行为保持不变，
所以多层版相对旧版的变化还包括这一缺测处理，不能将所有性能变化归因于跳接。

共享解码器每个融合阶段使用两层 3×3 Conv + GroupNorm + GELU。
跳接为投影后相加；主路径先 1×1 降维再上采样，控制高分辨率计算成本。
Adapter 为 `Conv1×1 → GELU → Conv3×3 → GELU → Conv1×1`，
最后投影的 weight/bias 均零初始化；每个 64→16→16→64 Adapter 有 4,448 个参数。
第一步只有末层投影收到非零更新，后续梯度会进入前两层。

交互使用本文下方已有的同步消息公式，区别是固定的 `residual_scale`
变为三个独立的可训练 `alpha_t`（初值 0.1，无硬截断，允许学到零或负修正）。
每条边仍有独立 1×1 消息投影和接收方/发送方拼接得到的空间标量 gate。
`mode=none` 时不创建消息、gate 或残差强度参数。

### 权重更新与调度

新增配置使用 `optim_wrapper.paramwise_cfg.recipe="official"`。
层编号与 rslearn 的 `LayerDecayAdamW.get_layer_id` 对齐：

- patch embeddings 和 composite encodings 为第 0 组；
- 第 i 个 block（从零开始）为第 i 组；
- encoder 最终 norm、其他 encoder 参数与全部下游模块为第 12 组。

`lr_i = 1e-4 × 0.65^(12-i)`：输入投影和第一个 block 为约 5.688e-7，
最后一个 block 为 6.5e-5，最终 norm 和下游模块为 1e-4。
不增加冻结预热或局部解冻；固定编码的原有 requires_grad=False 保留。
仅用于预训练 instance-level 目标、下游不执行的 `project_and_aggregate` 不参与优化。
输入 patch embedding 按预训练模型的可训练状态更新；v1.2 Base 默认可训练。
预训练目标分支不接入模型。

official 模式沿用 AdamW 的统一 weight decay（当前 0.05），不额外豁免
encoder bias/norm；只有 HydroState 自己的观测噪声参数保持无 weight decay。
这项 HydroState 扩展与官方没有观测噪声参数的模型明确区分。
旧配置默认为 `recipe="legacy"`，保持原来的分组、weight decay 豁免和调度行为，
不会改变既有实验或恢复训练。

新增配置保留 125 次迭代 warmup、50 epoch 训练和 BF16，batch size 16。
余弦下限使用 `eta_min_ratio=0.01`，保持各组比例；不能使用高于输入投影
初始学习率的统一绝对下限。这里不承诺最佳 batch size 或性能提升。

官方来源：
[微调建议](https://github.com/allenai/rslearn/blob/master/docs/foundation_models/OlmoEarth.md#fine-tuning-optimizer)、
[参数分组实现](https://github.com/allenai/rslearn/blob/master/rslearn/models/olmoearth_pretrain/optimizer.py)。

### 配置切换与监督边界

```bash
python tools/train.py configs/experiments/legacy/hydrostate_multilevel.py
python tools/train.py configs/experiments/legacy/hydrostate_multilevel_independent.py
python tools/train.py configs/experiments/legacy/hydrostate_multilevel_shared.py
```

使用已有的 `HYDROSTATE_MANIFEST`、`HYDROSTATE_MODEL_PATH`、`HYDROSTATE_TARGET_STATS`
配置数据和权重。也可直接覆盖 `model.encoder.out_indices`、
`model.adapter.bottleneck_channels`、`model.adapter.enabled`、
`model.interaction.mode`（none/sigmoid/softmax）、
`model.interaction.learnable_scale`、`model.interaction.residual_scale`。
新旧架构 checkpoint 不兼容；架构切换时使用预训练编码器开始新实验，
不要直接恢复另一种架构的完整训练 checkpoint。

三个新预设使用完全相同的优化器、标签支持和数据，适合比较 Adapter/交互贡献。
与旧模型比较时，要另外控制学习率分组、缺测池化、batch size 和监督口径的差异。

WF–SM 规则复用既有 `WaterSoilCoupling`：观测 WF 决定训练/验证聚合权重，
预测 WF 决定导出的 SM NoData；混合像元预测陆地部分 SM，纯水体不强制 SM=0。
ET 继续使用完整原生 PML 单元面积权重和既有标签时段。
没有新增 WF–ET 单调性约束，也没有将 ET 拆成没有独立监督的两个分量。

**重要数据边界：此提交实现上述网络和微调方案，没有凭空补齐 SMAP 足迹数据。**
现有 Zarr/manifest 只有窗口中心 SMAP 标量，因此新预设仍使用窗口陆地加权均值的
弱监督近似。完整足迹方案还需要观测 ID、日期、空间响应权重、覆盖检查以及按同一
观测分组的窗口采样；现有接口不具备这些信息，不能把 batch 内任意窗口拼成一个
SMAP 观测。当前粗尺度验证也不能证明 30 m SM 的细尺度精度。

### 验证

`tests/test_multilevel_model.py` 使用真实的官方小型 Encoder/Transformer 验证末层
结果、输入投影梯度、三个特征深度、checkpoint 重算、缺失模态与补齐时相的
batch 不变性，并验证三种预设的 40×40 输出、loss/predict 与 checkpoint 恢复。
`tests/test_layer_decay.py` 对照官方参数映射并检查余弦调度保持层间比例。
另外使用本地官方 v1.2 Base 权重，在 RTX 4060 上完成一次 120×120 三模态、
BF16 前向/反向/AdamW 更新与预测，确认输入 patch embedding 和编码器各深度
收到有效梯度。该检查使用合成标签，不是精度评估，也未提交 HPC 作业。

```bash
python tools/train.py configs/experiments/legacy/hydrostate_staged.py
python tools/train.py configs/experiments/legacy/hydrostate_staged_light.py
python tools/train.py configs/experiments/legacy/hydrostate_staged_independent.py
python tools/train.py configs/experiments/legacy/hydrostate_staged_softmax.py
```

也可在派生配置或现有 CLI 中组合数学选项：

```bash
python tools/train.py configs/experiments/legacy/hydrostate_staged.py --cfg-options \
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

## WF 约束 SM 的适用域和监督

`configs/experiments/legacy/hydrostate_staged_light_land_sm.py` 是独立的轻量版实验配置，启用
`model.water_soil_coupling`。原轻量版配置保持原数学定义，便于回溯和对照。
新增约束不增加学习参数，不改变 WF↔ET 或三任务的特征交换。

SM 定义为像元中适用土壤部分的体积含水量：混合像元保留 SM 本身，不能把
`(1-WF)*SM` 作为最终 SM；纯水区域输出 NoData。没有其他土地覆盖数据时，
非水体只是土壤适用面积的近似，不能据此声称已排除所有建筑、裸岩和冰雪。

默认配置：

```python
model = dict(water_soil_coupling=dict(
    enabled=True,
    supervision_source="observed",  # 也支持 "predicted"
    min_land_fraction=0.01,
    min_window_land_fraction=0.01,
))
```

- `observed`：训练和验证均采用有效的 30 m WF 标签计算权重。观测输入范围内
  若有缺测、非有限或超出 [0,1] 的 WF，该窗口不参加 SM 产品监督，不能将
  未知水陆范围重新归一化后假装覆盖完整。站点只检查自身足迹的完整性。
- `predicted`：训练和验证采用最终 WF 预测的 `detach()` 权重，辅助 SM 头
  复用同一份权重。只切断 SM 损失经面积权重回到 WF 输出的路径，原来的
  特征交互及共享主干仍可传递梯度。该模式的有效观测范围可能随预测变化，
  不建议用它代替默认的固定观测支持来作主要精度对比。
- 两个模式的 **30 m 输出掩膜始终来自预测 WF**，包括带标签的验证阶段。
  不把 WF 真值用于推理掩膜。`min_land_fraction=0.01` 表示陆地比例必须大于
  1% 才输出 SM；这是可调工程阈值，不是经过验证的科学阈值。
- 对通过像元阈值的有效陆地，窗口内使用
  `sum((1-WF)*SM) / sum(1-WF)`。保留下来的陆地面积还须达到输入有效面积的
  `min_window_land_fraction`；纯水/陆地不足的窗口或站点跳过 SM 损失。
  最终头与辅助头共用一次准备的权重、分母和有效计数，归约采用 FP32。
- 可选 `soil_moisture_domain_valid` 必须已对齐到输出网格 B×1×H×W，用于排除
  已知不适用的非土壤区域；本配置不自动引入或推测新的土地覆盖数据。

`mode='tensor'` 和内部训练前向保持有限、未掩膜的三任务张量。`predict` 在
SM 无效位置写入 NaN，并附带 `soil_moisture_prediction_valid` 和
`soil_moisture_land_fraction`；原始张量不被原地修改。区域文件导出尚未实现，
后续导出器应把这一掩膜写为 NoData。验证先聚合未掩膜的 SM，再生成预测图，
因此默认观测支持上的误差不会因错误的预测水体掩膜而隐藏。

验证图明确区分预测 WF 掩膜与监督 WF 权重；默认模式下，细尺度预测图为空
但观测尺度指标仍有效是可能的，表示 WF 误判而不是标签泄漏。
新增指标 `soil_moisture_observation_count` 记录验证的有效产品观测数，
`soil_moisture_prediction_coverage` 记录输入有效像元中有 SM 输出的比例；
覆盖率不是置信度或精度指标。训练的 `valid_count_soil_moisture` 仍包含可选站点。

此配置沿用已有训练集均值/标准差作为**固定参考归一化尺度**，不把它们宣称为
新的陆地筛选样本分布统计。WF-SM 筛选改变了有效观测集合，对照实验应在共同
有效支持上重算指标，并同时报告覆盖/观测数，不能直接比较旧日志的 hydro_score。
它仍使用 1.2 km 窗口与中心 SMAP 标签的空间近似，**没有实现完整 SMAP 足迹聚合**。
站点标签也没有因本修改自动补充。

启动入口：

```bash
python tools/train.py configs/experiments/legacy/hydrostate_staged_light_land_sm.py
# 恢复原轻量版 SM 定义的消融：
python tools/train.py configs/experiments/legacy/hydrostate_staged_light_land_sm.py --cfg-options \
  model.water_soil_coupling.enabled=False
```

参数形状和 state_dict 键保持兼容，可加载原轻量版权重；监督数学定义已改变，
应在独立 work_dir 开始新实验，不应把新旧日志合并为同一个连续训练结果。

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

`scripts/legacy/slurm/24_submit_staged_training.sh` 复用现有的
`22_train_merged_l40s.sh → 12_train_l40s.sh → 11_train_hpc.sh` 启动链。
它直接使用已准备好的 `samples_merged_v1/manifests/merged_v1_n20000`，不重做数据准备，
先提交 `hydrostate_staged_smoke.py`（8 次更新、全量验证、限时 2 小时），再提交
依赖其成功完成的 `hydrostate_staged.py`（50 epochs、限时 96 小时）。均为单张 L40S、
batch=16、BF16；正式训练从预训练编码器重新初始化，不从 smoke checkpoint 续训。
smoke 同时核验新增分支、辅助头和交互模块的参数更新。

建议先冻结源代码并验证与基线一致的数据指纹和训练设置，再运行：

```bash
HYDROSTATE_CODE_ROOT=/path/to/frozen/code \
  bash /path/to/frozen/code/scripts/legacy/slurm/24_submit_staged_training.sh
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
## Multilevel retrieval with target-time queries

`configs/experiments/legacy/hydrostate_ecosystem_multilevel_temporal_hpc.py` extends the multilevel
spatial architecture to all sixteen ecosystem tasks. Its
`MultiLevelTemporalOlmoEarthEncoder` extracts blocks 4, 8 and 12 in a single
backbone pass and retains exact acquisition times and per-pixel masks at every
depth. A shared `TargetTimeQuery` pools each depth for a requested instant or
interval before `MultiLevelFusionDecoder` combines the resulting spatial maps.
The shared decoder retains channels 256/128/64 and shallow-to-deep skip features.

Each variable has an independent 64→16→16→64 residual `OutputTaskAdapter` and a
64→32→1 regression head on the 40×40 output grid. Sigmoid-gated directed task
interactions and their learned strengths are retained. For a recipient queried
at time t, all source adapters receive features queried at t too. Only the
requested recipient edges are evaluated; changing one output's requested date
does not change the other outputs through interaction messages. The nine
land-cover fractions share one query and are jointly softmax-normalized, leaving
eight query groups for sixteen outputs. Decoder, query and encoder weights are
shared across these groups; their computation is still performed per query.

The data, loss supports and frozen target statistics match the single-depth
ecosystem temporal run. WF/SM land-mask coupling is disabled because WF and SM
can request different dates; re-enabling it would require an aligned WF support
at the SM query time. The original multilevel optimizer's official layer-decay
recipe and relative cosine floor are retained. Thus comparison with the existing
single-depth temporal run also includes the optimizer recipe, adapters and task
interactions, and is not an isolated test of encoder depth alone.

Run `bash scripts/legacy/slurm/28_submit_multilevel_temporal.sh` on the HPC login node
after synchronizing the committed code and generating ecosystem statistics.
This submits an independent 50-epoch job and reuses ordinary directory Zarr
labels. The script records the commit and source hashes and keeps outputs under
`/scr/u/xiaoz/HydroState/work_dirs/ecosystem_multilevel_temporal_JOBID`.
