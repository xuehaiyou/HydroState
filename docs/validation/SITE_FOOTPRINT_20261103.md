# ICOS ET 足迹与七组候选容量

本轮落实 ICOS 足迹重建和 SM/ET 联合候选容量核查。原始资料沿用 FossFS 已解压归档，
不访问 SQLite，不改已有产品或站点划分，不启动正式联合训练。

## 实现与范围

- `icos-footprints` 从 canonical `*_ARCHIVE_L2` 同时读取 FLUXNET_HH、FLUXES、
  VARINFO_FLUXES、ANCILLARY 和 SITEINFO。两种时序均按当地标准时，用 SITEINFO
  固定 UTC_OFFSET 转 UTC，再按起止时刻一对一匹配；不按行号拼接。
- ET 仍使用 LE_F_MDS / QC 0、1，复用有效 80%、实测 50%、昼夜各 80% 的窗口检查。
  FFP 不合格时段移除后重算 ET 和全部覆盖率；LE 不作足迹权重，负的半小时 LE 不截零。
- 根据 USTAR 测量高度历史和无物种后缀的总体冠层高度重建 `zm = height − 2/3 hc`。
  同日优先总体 90th Percentile，其次 Mean；排除 Dead 和单物种记录。
  同日期、最高优先级仍有冲突时，只将该更新至下一次明确更新之间设为未知，不拒绝整站。
  日期区间从结束日生效，不向过去回填。冠层信息最多沿用 366 天，这是本轮保守的
  开发参数，并非网络统一标准；年度内作物变化仍是近似，需随扩量检查。
- FFP 使用实有 WS、WD、USTAR、V_SIGMA、MO_LENGTH、PBLH；不填补缺失参数。
  PBLH 是来源方估计，不能称作实测。显式采用 `z0=.15 hc` 检查 `zm>12.5 z0`，
  排除粗糙子层；其余范围遵循作者 FFP v1.42。估计的位移和粗糙度保留为模型假设。
- 来源 FOOTPRINT_FLAG 只接受 0 或 3；3 表示缺少同步合格 NEE，不自动否定 LE，
  仍需通过本项目的 ET 和 FFP 检查。1、2、缺失值不进入共同有效时段。
- FFP 公式与作者未修改的 Python v1.42 对照，明确风向为来向。对齐实际 UTM 图块，
  处理子午线收敛角和尺度因子；对半小时足迹按窗口内相交秒数平均。
  不平滑、不用通量大小加权。采用 10→5→2.5 m 中点积分至 30 m 输出网格，
  相邻精度的权重 L1 差 ≤0.01 才视为收敛；覆盖量减该数值裕量仍 ≥0.95 才接纳。
  数值裕量不是统计置信区间。FFP 拟合曲线全平面解析积分约 1.001569，先以此定义
  概率密度；这是全空间归一化，不能替代裁剪后覆盖检查。
- 支持缓存为 Zarr，包含绝对质量 `mass`、归一化 `weights`、实际时间区间及持续秒数。
  明确标记 `label_support_only_not_training_sample`，尚未配遥感输入的 ET 不启用正式
  训练 mask。遥感输入配对后还须检查落在有效预测域内的绝对质量仍 ≥95%。

`availability` 复用既有 SM 候选，按七组及固定划分统计独立物理站点、去重站点窗口、
30 天间隔容量上界。SM/ET 同站同窗在 union 中只计一次；上界不是已完成的分层随机
采样，也不是可训练 Zarr 数量。本轮 ET 仅覆盖 ICOS，不能解释成全部来源已核查。

## 证据

- [FFP 作者代码与许可](https://footprint.kljun.net/download_2.php)，Python v1.42。
  公式来源：[Kljun et al. 2015](https://doi.org/10.5194/gmd-8-3695-2015)。
  许可随适配实现保存于 [FFP_LICENSE](../../hydrostate/data/site/FFP_LICENSE.txt)。
- [ICOS FFP 输入工具](https://github.com/icos-etc/FFP-utils/blob/main/CPtoFFPinput.R)：
  测量高度、总体冠层信息及参数来源。本项目不复用其缺测插值或固定高度替代。
- [ICOS 2025 FLUXES 变量定义](https://meta.icos-cp.eu/objects/E0KJJERYMTDbg4PfW26xazhk)：
  PBLH 与 FOOTPRINT_FLAG 的含义。
- [ICOS L2 Brasschaat 元数据](https://meta.icos-cp.eu/objects/0fcFp3gVgYggZ4Hv3W9L_uvR)：
  CSV 的 TIMESTAMP_END 为 local non-DST，门户另提供 UTC 时间轴，二者不可混用。

## 运行与验收

最终计算版本 `422e0478dad09c0ea97b5b0c1b16156fafcfb140`，定义 `site_icos_ffp_v2`。
固定 release 为 `/scr/u/xiaoz/HydroState/releases/site_footprint_422e047/`。

| 作业 | 范围 | 状态 | 实际耗时 |
|---|---|---|---|
| 4385689 | 3 个训练站、2023 年，开发探测 | COMPLETED / 0:0 | 13 秒 |
| 4385731 | 41 个 ICOS 站，站点与输入可用期交集 | COMPLETED / 0:0 | 63 秒 |

首次探测暴露历史元数据冲突处理过于粗糙，修正并通过测试后冻结 v2。旧探测产物保留，
不能与 v2 混作一批。最终作业 `c_foss_amd / normal`，2 CPU、6 GiB、0 GPU，
最大 RSS 854,028 KiB。申请时限依据 13 秒 / 138 个探测日历窗外推 9,890 个日历窗，
加 10% 后取整为 18 分钟；该估计保守，实际大部分窗口在低成本参数检查时已被排除。
提交记录在 `/home/xiaoz/HydroState/submissions/site_ffp_icos_20261103/`。

### 实际结果

- 41 站中 37 站完成时序处理，4 站没有明确、带日期的测量／总体冠层高度：
  DK-Vng、FI-Sii、NL-Loo、NO-Hur。它们未被算作“没有 ET 观测”。
- 在元数据可处理的 37 站中，**5,241 个窗口**通过原始 ET 时间质量检查；
  共同有效时段通过同样检查的只剩 **37 个**。夜间覆盖与参数缺失／适用条件是主要限制，
  不能将 5,241 个直接计为可用 ET 足迹样本。
- 37 个窗口中，7 个图块覆盖不足 95%，5 个积分在当前最细步长仍未收敛；
  **7 站、25 个窗口通过足迹筛选并保存 Zarr**。
- 25 个候选均属于中心水体频率 **0%** 组；21 个为 train、4 个为 test，validation 为 0。
  同物理站至少间隔 30 天后的容量上界为 **19 个**：train 16、test 3。
  这些是候选与容量上界，没有改变固定划分，也没有执行正式随机选样。
- 候选日期为 2019-03-06 至 2025-09-14；未按已有产品的 2017–2024 范围裁切。
  图块捕获质量为 95.920%–98.664%；最低有效时长 80.990%、实测 51.563%、
  白天 80.298%、夜间 81.594%。均在减去数值积分裕量后仍满足 95% 图块门槛。

| 站点 | 划分 | 足迹合格候选窗 |
|---|---|---:|
| BE-Lon | train | 9 |
| BE-Maa | train | 3 |
| DE-Geb | train | 3 |
| FR-Gri | train | 2 |
| SE-Deg | train | 1 |
| UK-AMo | train | 3 |
| DE-RuS | test | 4 |

当前已完成来源的容量如下；**不代表四来源全部候选已处理，也不是最终训练样本数**。

| 中心水体频率 | SM 站点窗 | ET 足迹合格窗 | 联合 30 天容量上界 |
|---|---:|---:|---:|
| 0% | 156,905 | 25 | 43,188 |
| (0,20)% | 308 | 0 | 90 |
| [20,40)% | 0 | 0 | 0 |
| [40,60)% | 0 | 0 | 0 |
| [60,80)% | 0 | 0 | 0 |
| [80,100)% | 0 | 0 | 0 |
| 100% | 0 | 0 | 0 |

联合 911 个物理站点组、157,238 个站点窗口，30 天容量上界 43,278。
高频组仍有缺口，不能挪用 0% 组填满后宣称七组均衡。头部字段调查显示，AmeriFlux
仍有值得优先核查的高频站：US-DPW／US-HB4（20–40）、US-EKH／US-RRC（40–60）、
US-HB1（60–80，固定 test）、US-GL2（80–100）。它们具备若干湍流参数字段，但缺少
PBLH 字段，测量高度、有效参数时段等仍需核查；这只是补资料优先级，不是合格数量。

### 检查与产物

- 18 项相关测试通过：作者原版数值、风向与适用域、未截断质量、时长加权、时区、
  同时段覆盖、元数据优先级与冲突、年末 5 天 ET、真实 Zarr 写读和断点复用。
- 与作者原版在 121×121 网格上比较未归一化密度，最大绝对差约 `1.36e-19`。
  作者 ZIP SHA256：`50e188c51eb84e18896cf0cac39fcd504ddc32d91afca4582352dd1c299864ba`。
- 25 个实际足迹 Zarr 全部通过读取、形状、有限非负权重、归一化、绝对质量、
  数值裕量、时长与 ET 标签一致性检查。Ruff、CLI 与 shell 语法检查通过。
- 正式 ET 输入配对数量仍为 **0**；缓存没有卫星输入，不是完整训练 Zarr。
  FFP 是带假设的足迹估计，不能称为直接观测验证的真实足迹。

HPC 产物根目录：`/fossfs/xiaozhen/HydroState/site/footprints_icos_20261103/`：

```text
config.json / report.json / candidates.parquet
stations/<station>/report.json / windows.json / footprints/<id>.zarr/
availability/availability.json / frequency_availability.csv
```

本地报告副本：`/mnt/d/Hydrostate/site_operations/20261103_footprint/result/`。
下一阶段按固定划分与 30 天间隔筛选候选、匹配五槽输入，再检查有效输入域中的足迹质量；
同时优先补高频 AmeriFlux 的缺失参数。输入未通过前不发布 SCR 或启用正式 ET 监督。
