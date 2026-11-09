# CoSA：共享状态与条件专家组合

CoSA 是可通过配置选择的新架构，现有 MTLoRA/shared-LoRA 行为和 checkpoint 保持不变。
当前只训练已有七个输出，新增变量训练流程和迁移效果尚未验证。

## 架构

```text
五个 8 天槽：S1 / S2 / Landsat + 时间、有效掩膜、观测质量
  → OlmoEarth v1.2 Base + shared-LoRA（一次主干遍历）
  → 第 4/8/12 层时相特征（同网格，不是三个空间分辨率）
  → 各层 LayerNorm + 768→256 投影
  → 共享四头目标时间注意力汇聚，各深度独立产生权重
  → 三层通道拼接 768→256 + 两层 3×3 空间残差融合
  → Coverage / SM / ET 独立空间专家（256→64 + 空间残差）
  → 可关闭的共享 Q–K Router（一轮有向交互）
  → 分组上采样/细化 + 30 m 任务输出头
```

三个专家组：Coverage 输出 water/snow_ice/land；SM 输出 soil_moisture；
ET 输出 et_water/et_ice/et_land。各输出激活和监督定义沿用公共配置。
专家分支合并 Adapter 与低分辨率空间解码职责，不额外建立 Shared Adapter。

## 特征前端和参数策略

Base 主干为 12 个 Transformer block、768 通道，输出索引 `[3, 7, 11]`。
从原始预训练 bundle 初始化，shared-LoRA rank=8，private_rank=0。
训练共享 LoRA、参与输入的 patch embedding、参与前向的 LayerNorm，以及全部新增模块；
原始注意力/MLP、位置/时间编码和未用预训练模块保持现有 shared-LoRA 冻结策略。
backbone 为 eval mode，但需要反向传播；不能整体使用 `no_grad()`。

CoSA 不安装额外 Landsat TOA 卷积，FM 原有 Landsat patch embedding 仍保留并训练。
旧架构默认仍安装该卷积，保持已发布权重的结构与行为。

### 已核实的 Landsat 数据边界

2026-11-09 核对仓库和原 20,000 样本库的四个真实分片后，发现它们实际声明并生产的是
**Landsat L1→TOA**：`landsat_l1_toa_native_mean_v2`、C2 T1 来源、
`toa_physical`、11 波段（含 15 m B8 与 B10/B11）。生产路径实际调用
`ee.Algorithms.Landsat.TOA`。这是样本元数据及代码证据，不仅是模块名称。
S2 则是 SR。此前“当前 Landsat 已是 SR”的讨论前提未被这些证据支持。

本次移除额外适配卷积，不修改已发布数据、归一化或生产签名，也不将 TOA 改名为 SR。
新配置严格复用当前真实输入契约。若另有 SR 样本库，需要独立核对波段、单位、
FM 入口及新的生产契约；不得直接传入本配置冒充现有 11 波段数据。
证据和抽查样本见 [本次验收](validation/COSA_20261109.md)。

## 时间与质量

各层保留 `B,T,C,H,W` 特征、`B,T,H,W` 有效掩膜和真实观测参考时间。
时间池化按空间位置进行，无空间全连接注意力。绝对时间编码包含年周期、日周期和
目标区间长度；相对时间包含观测与区间首尾及中心的距离。
复用现有五槽观测次数/时间分布摘要，作为 query 条件并直接进入共享状态。
即使只有一个有效观测，状态仍保留目标区间条件。

归一化/投影前屏蔽缺测值；全无效位置的状态及空间专家输出为零，并保留独立有效掩膜。
未知/缺测不会因 Query 或卷积偏置变成有效观测。最终输出仍通过现有 prediction_valid
及 SM 独立掩膜表达有效性，原始数值张量中的填充值不能脱离掩膜使用。

默认全部任务共用一个 `query_groups` 组，一次汇聚三层；区间不一致时明确报错。
需要不同输出日期时，显式配置任务分组。同组日期必须相同，覆盖 softmax 组不可拆分。
每次计算来源专家均使用接收方的查询区间，不使用来源任务在其他日期的输出。
投影和观测质量只在当前 forward 内缓存，不跨批保存计算图。

## 条件路由

设共享状态为 Z，专属特征为 E_i，任务嵌入 u_i 为 32 维：

```text
Q_i = W_Q[Z, u_i]
K_j = W_K[stop_gradient(E_j), u_j]
g_{j→i} = sigmoid(dot(Q_i, K_j) / sqrt(64))
E'_i = E_i + gamma_i * mean_j(g_{j→i} * W_V stop_gradient(E_j))
```

Q/K/V 共享，消息保持 64 通道。点积使用 FP32 保证混合精度稳定。
每个接收专家的 gamma 初始为零，Q/K/V 正常初始化。
第一步主要更新 gamma，随后 Router 获得任务梯度；不同时将 Value 投影初始化为零。
Key/Value 两条来源特征路径均停止梯度；共享状态和 backbone 仍可接受接收任务梯度。

`router.sources` 可显式指定有向来源集合，不允许自身、重复或未知来源；默认接收其他专家。
按允许来源数量平均，不按 gate 和归一化，所以全部 gate 都小时消息也会减弱。
仅同时交互一轮，输出不依赖遍历顺序。没有来源时恢复私有基线。
本轮不加入稀疏损失，也不宣称 gate 是因果关系或可以保证无负迁移。

B/C 都构建相同 Router 参数，使公共模块权重和构建后的随机数状态一致。
B 将 Router 参数冻结并绕过计算；因此 B 中保留的 Router 权重不属于有效可训练参数。
默认种子沿用现有配置；同种子不等于跨硬件逐位相同的训练轨迹。

## 输出与监督

七个内部输出及派生总 ET、SM 陆地加权聚合、各变量独立掩膜、原始像元 PML 损失，
全部复用 [现有科学规范](RETRIEVAL_PIPELINE.md)，本轮不改变损失或数据划分。

`ET_total = WF * ET_water + IF * ET_ice + LF * ET_land`。
Coverage 三输出共同 softmax；SM 为陆地条件下含水率；ET 分支非负。
站点接口可复用，但不能把本轮产品样本短测称为独立站点精度验证。

## 实验：已有 A，新训练 B/C

**2026-11-09 更新：本轮正式 B/C 改为产品＋站点联合监督。** 使用
`configs/hydrostate_joint_independent.py` / `configs/hydrostate_joint.py`，原产品专用配置
仍保留。站点损失均为 2，选模站点权重为 2/3；两种来源抽样各 50%，站点按任务→空间组→日期
均衡。数据统计、实际数量与监督限制见 [站点联合方案](SITE_SAMPLE_PLAN.md#2026-11-09cosa-联合训练)。
B/C 继续只切换 Router，使用相同有效 batch 和总优化步数。
A 的产品-only 监督与 B/C 不同，A 对 B/C 不能单独证明架构收益；严格架构比较需要增加
同样联合监督的 shared-LoRA 对照，本轮仅授权两次 CoSA 正式训练。

| 实验 | 配置/模型 | 初始化 |
|---|---|---|
| A | 已有 shared-LoRA 最佳验证 checkpoint | 原训练产物，只统一重新评估 |
| B | `configs/hydrostate_cosa_independent.py` | 原始 OlmoEarth；重新初始化 LoRA 和下游 |
| C | `configs/hydrostate_cosa.py` | 与 B 相同；仅开启 Router |

B/C 一次联合训练，不追加第二阶段。两者公共初始权重、数据顺序、有效 batch、
优化器、步数、选模与评估规则一致；禁止加载 A 的任务训练权重作为 B/C 的初始化。
公共默认设置为 AdamW lr=1e-4、weight decay=.01、50 epochs；有效 batch 以正式 GPU
显存探测确定，并在 B/C 中保持一致，不用默认 microbatch=2 代替显存探测。
联合 epoch 有 43,266 次有放回抽样，不能视为与原产品-only epoch 相同的训练预算。

A 对 B/C 是完整方案比较（包括共享状态和去掉输入适配卷积）；B 对 C 是交互消融。
旧权重需按原始输入路径评估，不能删除旧 checkpoint 的模块制造“同结构”。
同一测试集只做最终评估，checkpoint 按同一验证指标选取。
记录逐任务 R²、RMSE、有效监督量、负迁移、训练/推理速度、显存和参数量。
B/C 画精度随优化步数/墙钟时间变化；A 日志齐全时才加入收敛曲线。
两次单种子训练不证明跨随机种子稳定性，也不证明新变量迁移。

## 运行入口

数据和原始 FM bundle 路径仍通过环境变量设置。两次训练使用不同 work_dir：

```bash
python tools/train.py configs/hydrostate_cosa_independent.py
python tools/train.py configs/hydrostate_cosa.py
python tools/test.py configs/hydrostate_cosa.py CHECKPOINT.pth
```

`scripts/train.sh` 使用 `HYDROSTATE_CONFIG` 选择；Slurm 入口增加
`cosa_independent` 和 `cosa` 两个选项。区域推理仍用 `tools/predict_region.py`，
须提供匹配的 CoSA config/checkpoint，沿用现有拼接和导出。

真实样本短测（不是正式训练或最大 batch 探测）：

```bash
python tools/benchmarks/retrieval_model.py configs/hydrostate_cosa.py \
  --data-root /path/to/verified-subset \
  --pretrained /path/to/OlmoEarth-v1_2-Base \
  --samples 4 --batch-size 1 --device cuda --output /path/to/smoke.json
```

工具只读取训练子集、沿用原训练统计，不发布新统计或 checkpoint。
在 HPC 使用 debug 队列进行短测和目标 GPU batch 探测。正式作业单卡、适量 CPU，
明确覆盖 Slurm 脚本的历史资源默认值；walltime 按实测工作量加 10%，不能直接沿用 96 小时。
FossFS GPU 作业使用 c_foss_gpu。代码实现不等于正式 B/C 已提交或训练完成。

## 代码位置

- `models/cosa/encoder.py`：复用共享 LoRA，返回一套深度序列，不安装 TOA 适配卷积。
- `state.py`：共享目标时间查询和三层融合；时间编码复用 `models/query_time.py`。
- `branches.py`、`routing.py`：空间专家、上采样细化和 Q–K 交互。
- `model.py`：组装、query 分组、公共产品损失和推理接口。
- `configs/_base_/retrieval.py`：公共数据、监督和训练预算；旧配置解析值保持不变。
- `tests/test_cosa.py`：真实小型 OlmoEarth 的前向/反向、配对初始化、掩膜、时间、梯度和区域加载。
