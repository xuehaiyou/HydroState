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

运行记录、实测耗时及完成数量将在 HPC 实采后补充。规则先在固定训练站点检查，
再冻结用于全体 ICOS；不根据验证／测试集表现调整规则。
