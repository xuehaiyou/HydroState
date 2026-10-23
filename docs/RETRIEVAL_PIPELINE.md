# 当前产品监督流程（2026-10-23）

当前入口是 `configs/data/collection/retrieval.py` 和
`configs/hydrostate_retrieval.py`。采样、读取、训练和验证使用同一套新契约，
不自动转换旧的 4 槽或 16 变量样本。站点样本暂不接入。

## 时间与影像输入

- 目标日期覆盖 2017–2024 年，以 GEE 实际存在的 PML 日期为准。
- 输入为 **5 × 8 天**，第 3 槽对应目标 PML 周期；左右各两槽。
  年末 PML 周期不足 8 天时，监督保存实际截止日期。
- 每个图块覆盖 1.2 km × 1.2 km。采集存储保留产品分辨率：S1 10 m、
  S2 10/20/60 m、Landsat 15/30 m；本地读取时统一到 120 × 120、10 m，
  模型输出为 40 × 40、30 m。波段分组及网格定义见 [数据契约](DATA_SCHEMA.md)。
- 先逐景 QA，同一 UTC 日期的重叠影像先平均，再对不同有效日期等权平均。
  像元没有有效观测时填零并标记无效；零填充值不参加均值。
- 保存各传感器的各分辨率组、各槽、各像元的有效日期数、平均/最早/最晚日期偏移，
  并保存源影像 ID。日期统计经独立 `CompositeQualityFusion` 输入模型。
  `observation_time` 是槽中点参考时间，不能当作真实单景获取时间。
- 光谱、掩膜和日期摘要按相同像元中心最近邻索引对齐，不提前下载插值后的粗波段。
  同组光谱使用共同有效掩膜；组间允许不同观测日期，保留独立摘要而不合并日期数。
  模型对同一传感器采用各组有效掩膜交集，缺任一组的像元置零并标为无效。
- 默认至少两个槽中有一种传感器覆盖图块面积的 50%；不要求三传感器全部齐全。
  中心时间槽两侧使用未来影像，因此这是回溯反演，不是实时预测。

| 数据 | QA 与合成 |
|---|---|
| Landsat 8/9 C2 T1 L1 | `QA_PIXEL` 排除 fill、dilated cloud、cirrus、cloud、shadow；`QA_RADSAT=0`；保留水和雪。先按逐景系数转 TOA 反射率/亮温，再取均值，不能直接平均 DN。 |
| Sentinel-2 SR Harmonized | SCL 排除 0、1、3、8、9、10；保留水体 6 和冰雪 11。 |
| Sentinel-1 GRD | IW、10 m、VV/VH，使用产品有效掩膜；同一样本选择同一相对轨道和升降轨方向，dB 转线性功率平均后再转 dB。不用低后向散射阈值剔除水面。 |

Landsat TOA 使用显式物理尺度归一化和可训练的 11 通道输入适配器；
S1/S2 保留预训练归一化。新输入配置名为 `landsat_l1_toa_native_mean_v2`，
不宣称平均后的 TOA 输入仍等同于预训练 L1 DN 分布。

## 八组中心点均衡采样

七个 JRC-GSW 1.4 `occurrence` 组加一个独立 DW 冰雪组，形成八个**采样来源组**。
JRC 边界为 `0`、`(0,20)`、`[20,40)`、`[40,60)`、`[60,80)`、`[80,100)`、`100`；
冰雪组编号为 7。20,000 个成功样本每组 2,500 个；160 个试采样本每组 20 个。
采样组与真实 JRC 频率分别保存；冰雪组可能落在任一 JRC 组或 JRC 无观测区，
因此不宣称最终全部样本的七频率直方图也均衡。无需 MODIS。

1. **JRC 候选池**：打乱全球 2° 块，分批 `stratifiedSample`，缓存坐标及属性。
   未完成组积累剩余配额 3 倍的轻量候选后混合抽取，耗尽后补充。
   七组可先开始下载，无需等待冰雪池填满；JRC 无观测区不伪装成 0% 水体。
2. **独立冰雪候选池**：一次查询一个 0.5° 小区域和一个实际 PML 目标周期，
   用同窗 DW 至少一次 `P_I ≥ 0.60` 提出候选，位置和目标日期一并缓存。
   当前已采样中的雪冰区域提供搜索线索，同时持续搜索独立打乱的全球区域，
   不用 JRC 有效范围限制冰雪资格。每次最多 8 个候选，缓冲上限目标为 16；
   空结果也缓存。候选日期不因年月均衡而改动，其他日期需重新查询。
3. **先筛选，再下载**：按正式标签规则计算同窗覆盖，冰雪组要求中心 30 m
   像元有效且平均冰雪比例 ≥50%；这是覆盖比例，不是单景概率阈值。
   不合格直接拒绝，避免下载五槽完整输入。候选粗筛通过不代表最终一定合格。
4. **统一调度**：一个线程池最多 4 个任务，冰雪查询占其中一个名额。
   JRC 元数据补池前等待在途工作结束，不叠加另一组并发。两 CPU、零 GPU。
   JRC 日期在提交时选择；年份、月份、气候区按已完成加在途数量软均衡，
   地理块及 train/validation/test 数量作为进一步软排序，均不保证固定占比。
5. **成功配额与去重**：只有通过 QA 和双 Zarr 写盘的样本占配额，失败补采。
   位置/日期约束在八组间共享，同地点同日期只计一次。优先新地点；同地点
   最多四日期、间隔至少 40 天，附近重叠图块也限制近日期重复。
   网络错误保留断点并报错，不记成永久无标签。
6. **全图块监督**：组别只控制中心点抽样，图块内所有满足各变量质量条件的
   像元都参加监督，不按中心所属组剔除其他覆盖类型。

同地点固定地理划分；5° 划分块边缘保留 35 km 缓冲，避免粗标签支持范围跨划分。
八组硬配额针对整个成功样本库，各划分不要求八组等量。随机抽取针对候选点，
不声称全球像素按面积等概率入选。报告分别记录 `sampling_group_counts`、
`frequency_counts`、年月、气候、地理块、划分、每标签有效样本/像元数和冰雪查询成本。
小试需检查冰雪 `Ew` 实际有效监督量及空间划分；完成覆盖配额不等于可以训练全部分支。

## 水、冰雪、陆地标签

逐景读取 Dynamic World 九个概率；水体使用 Top-1，陆地概率为其余七类之和。
按下表分类，不把所有未达到水冰门槛的观测自动归为陆地。

| 类别 | 判定 |
|---|---|
| 水体 | 水体为九类 Top-1，且 `P_W ≥ 0.35` |
| 冰雪 | `P_I ≥ 0.60` |
| 陆地 | 其余七类概率之和 `P_L ≥ 0.85`，且 `P_W < 0.10`、`P_I < 0.10` |
| Unknown | 有效观测但未满足以上规则，不参加覆盖比例监督 |
| 云掩膜、缺测 | 没有有效观测；与 Unknown 分开统计 |

同一天重叠影像合并后，按有效日期平均，再按面积聚合到 30 m。
在有效覆盖域，`WF + IF + LF = 1`。云、缺测及 Unknown 观测
不参与覆盖均值；无效面积仍计入质量筛选的总面积分母。
30 m 像元已知覆盖面积必须 **>95%** 才监督覆盖比例。

这里的陆地是“非水、非冰雪”，包括建成区和裸岩，不能解释成全部具有可测土壤的面积。
这些概率阈值是质量筛选规则，不等于经过独立验证的准确率。
额外保存 `cover_observed_days` 与 `cover_unknown_fraction`，区分有效但分类不确定的观测
和云/缺测。Unknown 只屏蔽对应观测，不自动删除整个图块。
空间已知面积 >95% 不代表 8 天内 >95% 的观测都满足分类门槛；
覆盖均值依赖被保留的有效日期，Unknown 比例较高时仍可能有时间选择偏差。
有效日期数、时间跨度和 Unknown 比例用于分组审计；目前不额外剔除快速转换样本。

## 输出、掩膜和监督尺度

模型有七个内部输出：WF、IF、LF、SM、ET_water、ET_ice、ET_land。
三个覆盖比例用同组 softmax，SM 为陆地条件下的土壤湿度，ET 分支非负。
推理使用预测覆盖比例得到：

`ET_total = WF × ET_water + IF × ET_ice + LF × ET_land`。

ET 分支表示相应覆盖表面的 8 天有效蒸发/升华/蒸散率，单位 `mm/day`；
该合成不提供逐日过程或证明快速转换时段的过程精度。
总 ET 是派生输出，没有另造一个 PML 混合总量标签。

| 变量 | 标签和掩膜 | 损失尺度 |
|---|---|---|
| WF / IF / LF | 同窗 DW；每变量保存有效掩膜 | 30 m 覆盖比例 |
| SM | SMAP Enhanced L3 上午、`retrieval_qual_flag_am=0`；有效日期等权均值；保存实际有效日期 | 单图块陆地加权平均对单个 SMAP 值，`tile_mean_weak` |
| ET_water | PML `Ew`；完整原始像元内可信水体面积 >95% | 30 m 输出按真实相交面积聚合到完整 PML 像元 |
| ET_ice | PML `Ew`；完整原始像元内可信冰雪面积 >95% | 同上 |
| ET_land | PML `ET`；完整原始像元内可信陆地面积 >95% | 同上 |

PML `ET/Ew` 缩放为原值 ×0.01。各 ET 分支保存独立的 native value、valid、
area weights；混合明显、缺测过多或跨图块边界的不完整 PML 像元不参与 ET 损失，
但不会删除整个图块或影响覆盖变量监督。面积定义依据
[PML 调查](PML_SUPERVISION_AUDIT_20261022.md)。

SM 同时检查 SMAP 在 GEE 中发布的网格（产品标称 9 km）和以该像元中心为圆心的约 33 km 直径支持区：
两者都要求已知覆盖面积 >95%，水冰加未知面积 <5%；图块本身可信陆地面积也须 >95%。
实测 GEE v005 网格为 EPSG:4326、0.08°，不能假定它仍是原始 EASE-Grid。
33 km 圆仅为质量筛选近似，不是精确天线响应。SM 不复制成 30 m 真值，也不采集覆盖
整个 SMAP 支持范围的图块组；单图块对粗值仍然是**弱监督假设**。
地形异质性仅降低 SM 权重；默认 SM 损失权重为 0.25。

覆盖、SM 的样本/空间掩膜、各 ET 原始像元掩膜相互独立。
训练用观测水冰标签确定 SM 土地区域；推理只用预测覆盖与输入有效掩膜，
无需提供观测标签。站点独立验证留待后续，见
[站点调查](SOIL_MOISTURE_SITE_AUDIT_20261022.md)。

## 文件组织与执行

| 文件/目录 | 职责 |
|---|---|
| `hydrostate/data/retrieval/contract.py` | 唯一新数据版本、变量和数组形状 |
| `specs.py` | 产品、输入波段和标签的统一规格定义 |
| `label_packs.py` | 不重采输入的独立标签扩展、关联和审计 |
| `sampling.py` / `candidates.py` / `scheduling.py` | 配额、时空去重、候选池、逐任务日期选择与并发预占 |
| `catalog.py` / `ice_candidates.py` / `collector.py` | 元数据缓存、绑定日期的冰雪候选、采集及续采 |
| `composites.py` | QA、逐日去重、均值和观测质量统计 |
| `alignment.py` | 读取时统一 10 m，按相同索引对齐光谱、掩膜和日期 |
| `gee_labels.py` / `labels.py` | 产品读取与支持范围、独立监督掩膜、原始网格面积权重 |
| `validation.py` / `audit.py` / `datasets/retrieval_dataset.py` | 数组检查、发布审计与训练读取 |
| `models/composite_quality.py` | 观测次数/时间摘要输入模型 |
| `configs/hydrostate_mtlora.py` / `hydrostate_mtlora_shared.py` | MTLoRA / shared-LoRA 对照 |

```bash
# 预览不会连接 GEE 或创建数据目录。
bash scripts/run_gee_sampling.sh --dry-run

# 需要已授权的 Earth Engine 项目。修改配额等参数应使用新输出目录。
bash scripts/run_gee_sampling.sh
export HYDROSTATE_DATA_ROOT=/mnt/d/Hydrostate/samples/retrieval_5x8_eight_groups
export HYDROSTATE_MODEL_PATH=/path/to/OlmoEarth-v1_2-Base

# 默认检查全部分片契约及 64 个样本的数组；0 表示全部数组。
python tools/validate_zarr.py "$HYDROSTATE_DATA_ROOT" --pixel-limit 64
python tools/data/compute_target_statistics.py --config configs/hydrostate_mtlora.py \
  --output "$HYDROSTATE_DATA_ROOT/target_stats.json" --num-workers 4
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_mtlora.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_mtlora_shared.py
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/hydrostate_mtlora.py CHECKPOINT.pth
```

采样目录记录冻结配置和生产代码哈希、候选/目录缓存、逐样本成功或拒绝记录、
Zarr 输入/标签、`samples.parquet` 和 `collection_report.json`。
八组配额全部满足后才写 `_COLLECTION_SUCCESS.json`；训练准备再执行分片审计。
相同配置可续采；配置或生产代码变化需要新目录，不能混入旧标签定义。

统计量仅由训练划分计算并绑定数据契约。MTLoRA 在所有编码块使用共享 LoRA，
在第 4/8/12 层提供各任务专属出口；三个层级经任务解码器融合。
共享对照设置 `private_rank=0`。默认训练 patch embedding、LayerNorm 和新增模块。
两者数据、掩膜、统计量、训练轮数和有效 batch 一致。
单 GPU、最多 6 CPU；暂定 microbatch=2、累积=8，不能将其说成已测定的最优吞吐配置。

验证分别报告总体、八采样组和实际 JRC 频率组的指标（无观测为 -1），SM 使用相同弱监督尺度，ET 使用同样原始像元尺度。
产品一致性结果不能解释为独立的 30 m 反演真值精度；组间样本量与有效监督量应同时报告。

## 实测与实现状态

[阈值与完整样本试采记录](validation/RETRIEVAL_PILOT_20261022.md) 保存本次小规模 GEE 证据。
[候选池与实时均衡试采](validation/RETRIEVAL_BALANCED_PILOT_20261023.md) 记录 4 并发
140 样本的实际速度、年月分布和监督不足。
[八组与冰雪试采](validation/RETRIEVAL_EIGHT_GROUP_PILOT_20261023.md) 记录第八组加入后的吞吐与有效监督量。
[实现状态](IMPLEMENTATION_STATUS.md) 单独记录回归结果、正式数据进度和未决项。
正式执行入口与 HPC 参数见 [DATA_PREPARATION](DATA_PREPARATION.md)。
