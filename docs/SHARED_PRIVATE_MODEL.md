# OlmoEarth Shared / Private LoRA 实验

模型通过 MMEngine 的 `model.type` 选择。新架构注册为
`SharedPrivateHydroStateModel`，编码器为 `SharedPrivateOlmoEarthEncoder`；
原有类、配置和默认模型继续保留。这里描述已实现的训练流程，不代表已经验证精度提升。

## 配置选择

| 配置（均位于 `configs/`） | 注册模型 / 实验 |
| --- | --- |
| `hydrostate_v1.py` / `hydrostate_l40s.py` | 原 `HydroStateModel`，共享解码器 |
| `hydrostate_staged.py` | `StagedHydroStateModel`，阶段分支和特征交互 |
| `hydrostate_multilevel.py` | `MultiLevelHydroStateModel`，多层融合与输出 Adapter |
| `hydrostate_multilevel_temporal_l2.py` | `MultiLevelTemporalHydroStateModel`，实际观测时间查询 |
| `hydrostate_shared_private.py` | 新模型，阶段 A：共享 / 专属参数联合训练 30 epoch |
| `hydrostate_shared_private_specialize.py` | 新模型，阶段 B：固定共享参数，指定任务精调 20 epoch |
| `hydrostate_shared_private_joint50.py` | 对照：共享 / 专属参数持续联合训练 50 epoch |
| `hydrostate_shared_private_independent.py` | 对照：无 Shared LoRA，Private rank=16，50 epoch |
| `hydrostate_sharing_ablation_shared.py` | 三组严格对照 A：Shared rank=16，系数固定为 1 |
| `hydrostate_sharing_ablation_shared_private.py` | 三组严格对照 B：Shared/Private 各 rank=8，系数固定为 1 |
| `hydrostate_sharing_ablation_independent.py` | 三组严格对照 C：每进程单任务，Private rank=16 |
| `hydrostate_shared_private_lai.py` | 冻结原有知识，只训练新增 LAI 分支 |
| `hydrostate_shared_private_lst.py` | 冻结原有知识，只训练新增 LST 分支 |

入口保持一致：`python tools/train.py CONFIG.py`，评估用
`python tools/test.py CONFIG.py CHECKPOINT.pth`。模型、优化器构造器、输入合同和
运行目录由所选 config 决定。新增配置以 `_delete_=True` 重建模型和优化器，
不会继承旧架构的特征交互模块或层级学习率策略。

## 参数如何学习

对任务 `t`、Transformer 层 `l` 的 Q、V 投影，实际执行：

\[
W_{l,t}=W_l^0+s\,\sigma(g_{l,t})B_l^SA_l^S+s\,B_{l,t}^PA_{l,t}^P.
\]

代码不拼接完整权重矩阵，而计算三个线性投影之和；梯度通过冻结的基础网络，
更新共享低秩矩阵、当前任务的专属矩阵和共享门控。每个任务每层一个门控，
该层 Q、V 共用门控；门控是任务常量，不是逐像元路由。门控可以接近零，
并非精确的硬关闭。要完全关闭共享通路，配置 `shared_rank=0`。

- 基础权重 (W^0)：OlmoEarth v1.2 Base，全部冻结，保持 eval 状态。
- Shared LoRA：12 个 block 的 Q、V 各一个 rank=8 更新，供三个任务复用。
- Private LoRA：每个任务每个 Q、V 各一个 rank=8 更新。
- `lora_scale=1.0`：直接乘在低秩更新上，不隐含额外的 `alpha/rank`。
- 初始化：A 随机，B 为零；初始 LoRA 不改变基础线性映射。初始门控为 0.5。
- 每任务独立：Landsat L2 投影、实际时间查询、多层融合解码器、预测头、NLL 噪声参数。
- 每次任务编码读取第 4、8、12 层，在任务查询日期 / 区间上汇聚后解码。

只存一份基础权重。由于不同任务的有效参数不同，三个任务需要三个编码过程，
不能将该实现理解成一次共享编码，也不是已脱离 FM 的小型 CNN。其节省主要来自
可训练参数和优化器状态；是否节省总显存、计算和时间需要实测。

`gradient_checkpointing=True` 时，重算显式捕获原任务上下文，防止所有任务完成
forward 后 backward 错用最后一个任务的 LoRA。优化器只收集 `requires_grad=True`
的参数；在一个优化步的所有累积批次都没有某任务有效标签时，该任务不会因
AdamW 的 weight decay 或历史动量被更新。共享参数仍可能被其他任务更新。

## 数据和物理关系

默认复用现有 `landsat_l2_sr_v1` 输入：S1、S2、Landsat L2 SR、有效掩膜和真实
观测时间。每个任务必须有 `target_time`，使用 UTC 秒的 `[t,t]` 或 `[start,end)`。
旧的 Landsat L1 输入配置仍按原模型工作，不能把 L1 / L2 checkpoint 直接互换。

WF 用 sigmoid，SM 用 sigmoid，ET 用 softplus；训练仍按各产品的观测支持比较：

| 任务 | 监督支持 |
| --- | --- |
| WF (`water`) | 10 m 标签完整有效时聚合到 30 m 输出 |
| SM (`soil_moisture`) | 有效窗口平均，对照现有 SMAP 中心标量标签 |
| ET (`et`) | 完整有效的 PML 原生像元，按面积权重投影 |
| LAI (`lai`) | 新标签包提供原生网格、面积权重和时间区间 |
| LST (`lst`) | 输出网格的有效像元、实际观测时刻；默认不作为输入 |

SM 仍是窗口近似弱监督，不是完整 SMAP footprint 监督。sigmoid 的 0–1 范围
不是土壤孔隙度约束，非水面也不自动等于有效土壤。

任务间首先通过 Shared LoRA 保持知识联系；默认 `water_soil_coupling=None`，
因为 WF 和 SM 标签日期可能不同。已有陆地条件 SM 支持规则可以在**实际对齐**
日期、网格与有效域后启用；模型会拒绝 WF / SM 查询时间不同的请求：

```python
_base_ = ['./hydrostate_shared_private.py']
model = dict(water_soil_coupling=dict(
    _delete_=True, enabled=True, supervision_source='observed',
    min_land_fraction=0.01, min_window_land_fraction=0.01))
```

训练的 observed 模式使用观测 WF；预测时使用预测 WF。支持权重 detach，
SM 损失不会反向推动 WF 修改掩膜。仅修改查询时间不能使不同日期的标签变成
对齐观测，数据准备必须先完成这一步。阶段 A 默认仍有共享梯度竞争；阶段 B
固定共享知识后独立收敛。没有加入缺乏通量与储量支持的水量闭合损失，
也没有强加 WF、SM、ET 单调关系或声称严格物理一致性。

## 三变量训练与组合

以下命令在已安装项目的 `hydrostate` 环境执行。先设置实际数据 / 权重路径：

```bash
export HYDROSTATE_MODEL_PATH=/path/to/OlmoEarth-v1_2-Base
export HYDROSTATE_DATA_ROOT=/path/to/samples
export HYDROSTATE_L2_PACK=inputs/landsat_l2_sr_v1
export HYDROSTATE_TARGET_STATS=/path/to/samples/target_stats_shared_private_l2_v1.json
# 与现有启动脚本相同：MMEngine 0.10 的完整 checkpoint 含日志对象。
# 此设置用于本项目训练生成的可信 checkpoint，adapter bundle 仍按 weights_only 加载。
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_shared_private.py \
  --output "$HYDROSTATE_TARGET_STATS"
python tools/train.py configs/hydrostate_shared_private.py
```

统计文件只使用训练集，绑定 manifest、输入和标签合同，拒绝覆盖已有文件。
每个配置有独立默认 `work_dir`；若设置 `HYDROSTATE_WORK_DIR`，需要自行为每次
实验指定不同目录。默认每卡 batch=2、累积 8 步；多卡时全局有效 batch 随卡数增加。

阶段 B 的三个任务均从**同一个阶段 A checkpoint** 出发，分别选择各自验证 RMSE
最优点，使用新优化器和学习率调度，不恢复阶段 A optimizer：

```bash
export HYDROSTATE_SP_CHECKPOINT=/path/to/stage_a/best_hydro_score_epoch_N.pth
for task in water soil_moisture et; do
  HYDROSTATE_SPECIALIZE_TASK="$task" \
    python tools/train.py configs/hydrostate_shared_private_specialize.py
done
```

`model.train_shared=False` 与 `model.train_tasks=['soil_moisture']` 即为精调控制。
冻结内容包括旧任务输入投影、查询、解码器、预测头和噪声参数。改变阶段后必须
重建 optimizer；配置启动会自动做到，不应跨阶段使用 `resume=True`。

用各实验保存的展开配置导出专属分支，然后组合到阶段 A 权重上：

```bash
python tools/export_shared_private.py /path/to/water_run/expanded.py \
  /path/to/water_best.pth outputs/water_v1.pth --tasks water
python tools/export_shared_private.py /path/to/sm_run/expanded.py \
  /path/to/sm_best.pth outputs/sm_v1.pth --tasks soil_moisture
python tools/export_shared_private.py /path/to/et_run/expanded.py \
  /path/to/et_best.pth outputs/et_v1.pth --tasks et

python tools/compose_shared_private.py /path/to/stage_a/expanded.py \
  "$HYDROSTATE_SP_CHECKPOINT" outputs/hydrostate_sp_v1.pth \
  --task-bundles outputs/water_v1.pth outputs/sm_v1.pth outputs/et_v1.pth

python tools/test.py /path/to/stage_a/expanded.py outputs/hydrostate_sp_v1.pth
python tools/export_shared_private.py /path/to/stage_a/expanded.py \
  outputs/hydrostate_sp_v1.pth outputs/knowledge_v1.pth
```

`expanded.py` 表示 MMEngine 在各 `work_dir` 保存的实际配置文件，替换为其真实名字。
组合后的 checkpoint 可用于三变量区域预测，并应再次验证整体产品。它没有
optimizer / scheduler 状态，适合推理或新阶段 `load_from`，不能用于精确续训。
同阶段恢复则使用原始 Runner checkpoint：

```bash
python tools/train.py /path/to/run/expanded.py --cfg-options \
  resume=True load_from=/path/to/run/epoch_10.pth model.adapter_init=None
```

adapter bundle 保存共享及指定任务权重，核对基础权重哈希、共享权重哈希、
rank、输入合同、输出尺度、观测支持、WF–SM 支持规则、分支结构和旧任务统计。
组合时不覆盖共享参数；共享版本不同、重复任务或不完整权重会报错。
导出和组合工具均拒绝覆盖已有产物。

## 新变量少样本适配

新增 LAI 时，先准备符合现有 variable-pack 合同的标签包，再计算扩展任务统计。
默认保留原 manifest / 输入 / 旧标签，新增变量只有部分样本有标签即可。
旧任务归一化统计必须与导入 bundle 完全相同，不能用少样本子集重新估计旧任务
的 mean / std。默认配置不自动重写旧任务统计；不兼容时会在加载阶段报错。

```bash
export HYDROSTATE_SP_BUNDLE=/path/to/outputs/knowledge_v1.pth
export HYDROSTATE_LAI_PACK=labels/ecosystem_v1
export HYDROSTATE_TARGET_STATS=/path/to/samples/target_stats_shared_private_lai_v1.json
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_shared_private_lai.py \
  --output "$HYDROSTATE_TARGET_STATS"
python tools/train.py configs/hydrostate_shared_private_lai.py
```

LST 对应 `hydrostate_shared_private_lst.py`、`HYDROSTATE_LST_PACK` 和独立统计文件。
两个示例均从三变量版本扩展；连续增加多个变量时，应派生新 config，声明完整的
`tasks`、`heads`、`label_packs`，导入包含所有旧分支的 bundle，并让 `train_tasks`
只包含新增任务。不需要改注册表、训练循环或旧模型代码。

新任务学习自己的门控、Private LoRA、输入投影、查询、解码器和头；Shared LoRA
及所有旧分支冻结。这是有标签的任务适配，不是零样本反演。四变量及更多变量可用
MMEngine test loop；当前区域拼接产品接口仍限制为 WF / SM / ET 三变量。

## 容量选择与对照

参数容量通过派生配置选择，例如：

```python
_base_ = ['./hydrostate_shared_private.py']
model = dict(encoder=dict(
    private_rank=dict(_delete_=True, water=8, soil_moisture=16, et=32)))
work_dir = 'work_dirs/sp_rank_8_16_32'
```

按验证集比较 rank=8/16/32；这不是自动 rank 搜索，也不支持把已训练 rank=8
分支原地扩成 rank=16 继续加载。新增任务可选自己的 rank，只要导入的旧任务 rank
保持不变。主实验比较 30+20 阶段训练、50 epoch 联合训练、无共享独立 LoRA，
以及原有联合微调。独立 LoRA 的 rank=16 匹配每任务活动的 8+8 容量，但不匹配
总存储参数量；同时报告总参数、可训练参数、显存、耗时和精度。

验证重点是每任务 RMSE / bias、空间与时间独立留出、各任务是否负迁移，
以及新变量在相同少样本划分下相对独立 LoRA 的收益。不同旧配置的 L1/L2 输入、
数据划分、有效域、预算不同，不能将差异全部归因于参数共享方式。

## 本地验证

`tests/test_shared_private_model.py` 使用官方 OlmoEarth 的小尺寸网络，无需下载权重，
覆盖参数公式、零初始化、checkpoint 重算、基础参数冻结、缺标标签、单任务精调、
LAI 扩展、版本拒绝、分支组合、MMEngine 训练与恢复、双进程 DDP，以及配置隔离。
这些测试验证实现与数据接口，不替代真实预训练 Base 网络的完整实验。

2026-10-19 本地验证：全仓库 CPU 回归 321 项通过，1 项 GPU 用例跳过；
新架构的 13 项测试在可用 CUDA 的环境全部通过，包括 BF16 AMP 训练。

## HPC 两组实验提交

在固定 commit 的独立源码目录中运行 `tools/submit_shared_private.py`，指定唯一的
`--root`、`--data-root`、`--manifest`、`--statistics` 和 `--model-path`。
提交前会校验完整训练数据合同，复制冻结统计并记录源码校验和。

作业关系为：

```text
joint50（50 epoch 持续联合训练）
shared30（30 epoch 共享 / 专属联合训练）
  ├─ water（20 epoch，仅 WF 专属参数）
  ├─ soil_moisture（20 epoch，仅 SM 专属参数）
  └─ et（20 epoch，仅 ET 专属参数）
       └─ compose（三个分支都成功后，生成 staged_composed.pth 和 knowledge_v1.pth）
```

五个训练作业各申请一张 L40S、8 CPU、64 GB 内存和 96 小时；组合在 CPU 作业中完成。
三个专属分支使用同一个共享阶段验证 `hydro_score` 最佳 checkpoint，分别按自己的
验证 RMSE 选优。两组固定 seed=3407，每卡 batch=2、累积 8 步；按任务计算均为
50 epoch，但多分支运行的总计算量不相同。

`submission.json` 在每次 sbatch 前记录提交意图，收到结果后立即记录 job ID 和依赖。
同一目录拒绝重复提交；若提交中途失败，应检查记录及 Slurm，再补交缺失阶段。
现有作业和原实验目录不会被覆盖。`logs/` 保存各阶段 stdout/stderr，阶段目录保留
MMEngine 展开配置、训练日志、checkpoint 和初始化来源。

## 三组共享程度对照

`tools/submit_sharing_ablation.py` 提交三个模型组，各使用 seed=3407/3408/3409。
每个 seed 有 A（共享）、B（共享＋专属）、C-WF、C-SM、C-ET 五个独立训练进程，
总计 15 个 GPU 作业；一个 CPU 汇总作业以 `afterok` 依赖全部训练与评估成功。
这是三组实验，不是只提交三个 Slurm 作业。原有联合/分阶段实验保留各自配置。

三个模型组都冻结同一 OlmoEarth，训练 50 epoch、batch=2、累积 8 步，使用相同
数据、监督、NLL 和学习率规则。`shared_gate_mode='fixed_one'` 精确使用共享系数 1，
不创建可学习门控；旧配置默认仍为 `learned`，旧版 adapter bundle 按旧语义加载。
`branch_init_seed` 保证同一 seed 的输入投影、查询、解码器和头在三组中初值相同，
不因不同 LoRA rank 消耗随机数而改变预测分支初始化。

三组 BF16 均设置 `loss_scale=dict(enabled=False)`，避免单任务的整个梯度累积窗口
缺标时 GradScaler 因没有参数梯度而报错。空窗口保持参数和 AdamW 动量不变；之后
有标签的窗口正常训练。该设置仅作用于新的三组对照配置。

共享范围是 **FM 的 LoRA 参数**。所有组保留相同的专属输入投影、时间查询、解码器
和头，因此 A 不是整个模型、解码器和输出头全共享。A/B/C 每任务的 LoRA 秩上限
均为 16，但三组总 LoRA 参数量分别与 16/32/48 成正比。这轮没有提交等总参数量
复核或门控/梯度冲突消融，不能把主实验表述为等总参数预算实验。

C 配置使用 `train_shared=False`、`train_tasks=[task]`、`active_tasks_only=True`：
每个任务独立优化和裁剪梯度，只前向、计算损失和评估指定任务，其他任务的标签
不参与该进程的目标。三个任务仍读取同一 manifest 和完整统计合同，以保持采样顺序、
掩膜和归一化一致。为兼容通用 checkpoint，进程仍构建并冻结其他分支；统计报告区分
存储参数与实际训练参数，不把 checkpoint 文件大小当作精简部署模型大小。

提交示例（在已提交、干净的 HPC release 中执行）：

```bash
python tools/submit_sharing_ablation.py \
  --defer-on-quota \
  --root /scr/u/xiaoz/HydroState/experiments/sharing_ablation_RUN_ID \
  --data-root /scr/u/xiaoz/HydroState/samples_merged_v1 \
  --manifest /scr/u/xiaoz/HydroState/samples_merged_v1/manifests/merged_v1_n20000/samples.parquet \
  --statistics /scr/u/xiaoz/HydroState/samples_merged_v1/manifests/merged_v1_n20000/target_stats_shared_private_l2_v1.json \
  --model-path /scr/u/xiaoz/HydroState/pretrained/OlmoEarth-v1_2-Base
```

提交器先验证训练统计合同和原始 split seed=42 的 5° 地理块划分，再冻结 15 份展开
配置。运行节点同时验证源码、配置、manifest、统计、FM 权重及 L2 索引/完整标记
校验和。原生粗分辨率标签 footprint 的跨块重叠没有在此步骤穷尽审计，地理块检查
不能表述为所有标签绝无泄漏或独立年份泛化验证。

每个训练作业保留最终 epoch=50 和验证选优 checkpoint，随后在同一 GPU 分配内
运行 `tools/evaluate_sharing_ablation.py`。主结果是固定 epoch=50；补充结果是 A/B
按验证 `hydro_score`、C 各自按验证 RMSE 选优。测试集不参与选模型。两种选优策略
天然不同，补充表不能被解释成仅共享参数这一因素的差异。

每次评估输出 `evaluation/metrics.json` 和逐样本误差汇总 `sample_metrics.jsonl`，
保存样本 ID、有效观测数、误差平方和、绝对误差和等，可按冻结 manifest 关联空间。
参数归属、配置与 checkpoint 哈希、训练总耗时和评估显存也会记录；训练耗时包含
验证和保存，不是纯训练吞吐。训练显存与迭代时间仍见原 MMEngine 日志。

CPU 汇总生成运行根目录的 `comparison.json`，报告每任务 RMSE、种子间标准差、
标准化平均 RMSE、相对独立训练的收益，以及以相同种子和 5° 空间块配对重采样的
2000 次 bootstrap 区间。先核对测试样本与有效观测支持完全一致，再计算对照；
区间是探索性区间，未做多重比较校正。单个变量退步必须与综合指标同时报告。

`--defer-on-quota` 从 Slurm 读取 GPU QOS 的课题组/用户提交上限，并按展开的 array
任务计数。满额时由一个 normal QOS CPU 控制作业管理续提，每次检查后退出；若仍有
待提交实验，只创建一个最早 10 分钟后启动的下一次检查，不占 GPU 等待。仅在实际
队列显示空位时提交；额度竞争导致拒绝后保留原始返回，队列状态未变化时不重复尝试。
`submission.json` 中 `waiting_for_gpu_submission` 是尚未被 Slurm 接受的训练任务，
`active_dispatcher` 是唯一续提作业，不能把它们表述成全部训练已进入 GPU 队列。
全部训练作业接受后，才提交带完整依赖的汇总作业。文件锁和每次提交前后落盘防止
重复续提；非配额错误或不明确的 sbatch 返回会停止自动流程，供检查后恢复。
