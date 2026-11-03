# 站点流程首轮实施（2026-11-03）

状态：首批真实 SM Zarr 和 CPU 训练路径小试完成，完整流程仍在实施。
已授权按[完整方案](../SITE_SAMPLE_PLAN.md)推进；本记录区分实现、
合成测试和真实资料验收。未访问观测 SQLite，也未调整原入库进程。

## 已实现的边界

- `tools/data/prepare_sites.py inventory/extract`：ZIP 中央目录清点、路径校验、
  SHA256 与解压 CRC 检查、同级永久解压、嵌套解压、完成标记和断点跳过。
- `labels`：直接读取永久解压的 ISMN `.stm` 与标准 FLUXNET HH/HR CSV。
  保留逐日晨间、原文件 SHA256、深度和时间解释；ET 原始与校正版本分开。
  不同或缺失的历史时区／坐标、未知支持、非浅层 SM 明确记录 unsupported；
  格式错误和 I/O 错误向上传播，不冒充无有效观测。
- `pilot`：复用当前原生分辨率遥感输入生产，生成独立 `inputs.zarr/labels.zarr`；
  目前为明确标识 `pilot_unreleased` 的 SM 开发样本，不发布至 SCR。
- `SiteDataset`：独立校验站点版本、生产指纹、网格和形状；与产品共用对齐及翻转。
  点位和足迹分别存储、随增强同步变换；产品标签占位均为无效。
- 当前模型可显式启用 `site_supervision`，用训练站点统计计算独立 SM/ET 损失；
  不增加输出头，SM 不缩放 LF，ET 对三类率及预测比例保持梯度。
  这是底层监督入口；正式联合配置、混合采样、划分和统计绑定尚未完成。

当前未适配 AmeriFlux BASE 的细粒度质控及 ICOS 的跨产品时间／足迹元数据。
这些原始资料先上传、永久解压；不将不确定来源强行送入主标签。
标准 FLUXNET ET 即使时长质量通过，只要足迹或有依据的空间代理未确认，
仍是候选标签，不是有效 ET 训练样本。

## 运行与数据

- 原始包清单：ISMN 1、ICOS 1、AmeriFlux BASE 556、FLUXNET Shuttle 781，
  合计 1,339 ZIP、约 72.62 GiB；数量是压缩包数，不是站点数或训练样本数。
- 中央目录展开体积：ISMN 20.87 GiB、AmeriFlux 32.62 GiB、FLUXNET 191.35 GiB。
  ICOS 外层展开 5.05 GiB 仍是嵌套包，不能当作最终展开体积。
- 原始与永久解压位置：`/fossfs/xiaozhen/HydroState/site/raw/`，四个来源各自目录。
- 本地传输清单、逐来源日志与可续传状态：
  `/mnt/d/Hydrostate/site_operations/20261103/`。
- 首次同步代码 `7836c0df99ddf0e5a6387f7cb08d497b9ec2ff03`；冻结副本位于
  `/scr/u/xiaoz/HydroState/releases/site_7836c0d`。
- CPU 小试 `4382397`：`c_foss_amd`、debug、2 CPU、4 GiB，21 秒完成，ExitCode 0。
  US-CS5、US-Ne1 两包共 139,273,035 字节，展开 457,447,303 字节；
  本地与远端 ZIP SHA256 完全一致。
- ISMN 解压 `4382467`：相同资源，解压阶段 233.67 秒，20,105 文件，
  22,407,548,722 字节。ZIP SHA256：
  `6b455c061bf7b671835513c1d89f57a1acbe3090459974a7563aef2ae0cffe7a`。
- HPC 解压报告：`site/reports/extract_pilot.json`、`extract_ismn.json`；
  日志：`site/logs/extract_*_<jobid>.log`。完整队列状态需实时查询。
- ICOS `4382519`：6 分 34 秒完成，ExitCode 0；外层及 210 个嵌套包永久保留，
  不需要再次解压。AmeriFlux BASE 556 包上传完成，`4382570` 以 normal QoS
  解压，6 分 47 秒完成，ExitCode 0。
- FLUXNET 全量上传仍在继续。后台 `queue_after_transfer.py` 仅在对应 rsync 成功后
  提交该来源的 CPU 解压；状态写入上述本地运行目录的 `remaining_extraction_jobs.json`。
  时限基于已测约 95 MB/s，按 90 MB/s 与包级开销估算，再加 10% 余量。

## 验证

`python -m pytest -q tests/retrieval`：100 通过；Ruff 与差异检查通过。
随后加强站点数组边界校验，相关 Zarr 读写／增强测试重新通过。
覆盖时间单位、UTC 跨日、最近合格晨间、年末短窗口、未来上下文、昼夜／实测覆盖、
极昼、重复时戳、未知时区、浅层与非点支持、不可覆盖解压、独立掩膜、
SM 不受 LF 缩放、ET 混合梯度、站点批次的真实 U-Net CPU 前向和反向。

本地 Pandas 3 默认解析为微秒，而旧环境可能为纳秒；解析现显式转为秒。
测试使用缩小的 U-Net 验证计算路径，不等于已完成 GPU 资源或正式训练验收。
昼夜长度使用 [NOAA 日出日落近似](https://gml.noaa.gov/grad/solcalc/solareqns.PDF)，
按日计算赤纬／均时差并区分极昼夜，不宣称秒级天文精度。

后续新增 SMAP 真实 EE 计算图、配对日期、负 R²、JRC 有观测的零频率点等测试；
最新 `test_site*.py + test_gee_expressions.py` 为 20 通过。图形 PDF 字体最小 8 pt，
碰撞审计 0 FAIL / 0 WARN；单面板无需多面板对齐。PNG 为 600 dpi，另存 PDF/SVG。
另补充带站点 SM/ET 的 MTLoRA 小模型前向／反向回归，验证同一接口能够经过
共享／专属 LoRA 路径；该合成小模型测试不代表真实 FM 权重已完成站点 GPU 训练。

## 真实小试结果

1. CPU 标签作业 `4382496`，8 秒完成：Kerasia 57 个目标期中，54 个满足晨间覆盖门槛；
   US-CS5 的 46 个目标期中，28 个满足 ET 时长质控，但因足迹未确认，有效 ET 训练量仍为 0。
   这些是候选窗口，尚未按 30 天间隔全部抽样，不能作为最终训练样本数。
2. 原始记录核对：Kerasia 2021-10-08 至 10-16 的晨间均值为 0.170125 m³/m³，8/8 天；
   US-CS5 2021-07-12 至 07-20 的标准 ET 为 3.341156 mm/day，校正版本为 4.592312，
   有效时长 98.958%、实测时长 71.354%、白天有效 98.621%、夜间有效 99.540%。
3. 固定随机种子 43，从合格 SM 窗口抽取 6 期并执行同站至少 30 天间隔：
   2021-12-03、2022-02-10、03-14、05-09、10-24、12-19。没有依据与 SMAP 的一致性挑日期。
4. HPC IO 节点完成 6 份独立 Zarr，约 5.7 MiB；全部 SM 有效，ET 均无效。
   数据位置：`/fossfs/xiaozhen/HydroState/site/pilot/site_5x8_am_20261103_b/`。
   输入生产使用冻结代码 `693836e01640559ad9beab89a046e2fa4fc4d52f`，2 CPU、0 GPU；
   该目录标记为 `pilot_unreleased`，没有写入正式 SCR 训练目录。
5. 真实模型检查 `4382573`，9 秒完成：使用当前 U-Net 完整通道宽度、8,202,446 个参数，
   batch=2、120×120 输入、40×40 输出，前向／反向、有限且非零梯度通过。
   站点统计来自这 6 个开发窗口，SM 均值 0.122333、总体标准差 0.062062。
   执行代码 `f7c12ea73ce0d2bc0347e0c53967cf8b58a76f28`；本次没有优化器训练任务、
   GPU 验收或正式精度评估，不把该小试统计用于未来正式实验。

Kerasia 在本轮作为开发站点使用，后续正式划分需保留开发记录，不能再把它当作未见测试站。
只验证了一个 0% 频率组站点；没有验证七组配额、网络间身份合并或空间隔离。

## SMAP 一致性预检

使用项目 `SOURCES` 的相同 SMAP 版本（本批为 v005），从实际发布网格读取 AM 值与
`retrieval_qual_flag_am=0`，按共同日期分别重算站点和产品平均，不覆盖训练标签。
[GEE SMAP 目录](https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL3SMP_E_005)
没有暴露精确过境时刻，因此本图明确标记 UTC 日期近似。

6 个窗口中 5 个有共同有效日期，分别为 3、3、3、5、5 天；2022-03-14 无共同合格值。
配对 RMSE **0.102389 m³/m³**，Bias（产品减站点）**+0.099657 m³/m³**，
R² **−1.083500**。R² 使用 `1−SSE/SST_site`，没有用相关系数平方替代。
这些只有一个站点，且点对粗像元的空间代表性尚待核实，**仅为诊断，不是模型精度或
正式验收通过结论**。没有因为差异较大而删除站点或改变质量阈值。

HPC 产物位于上述小试目录的 `reports/product_consistency/`：
`sm_smap_scatter.png/pdf/svg`、`sm_smap_pairs.csv/json`、原产品响应缓存与图形审计。
JSON 保留逐日配对、源文件校验值、原训练值、QA／缺失原因和原生网格。
本地副本位于 `/mnt/d/Hydrostate/site_operations/20261103/site_5x8_am_20261103_b/`。
ET 空间支持还未接受，因此本次没有生成 PML 主对照图，也没有将校正 ET 混入主标签。

## 小试发现并修复的 JRC 零频率问题

该站 GSW occurrence 与 YearlyHistory 查询均为空，但
[Metadata](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_Metadata)
返回 `valid_obs=445`、`detections=0`、`total_obs=658`，MonthlyHistory 的抽查月份为非水。
这是有观测的从未检出水体点，不能解释为未观测。

站点流程现在用 `valid_obs>0` 确认可用性；occurrence 缺值时，只有 `detections=0`
才恢复为 0%，有水体检出却缺 occurrence 则拒绝并记录。该定义绑定
`frequency_observation_policy=jrc_metadata_valid_observations_v1`。
失败的第一版空小试目录与日志保留，使用新目录 `_b` 重跑，没有覆盖成功样本。
已有产品 Zarr 没有重写；其既有零频率候选空间是否受 YearlyHistory 掩膜限制需另行审计。

## 已支持的执行入口

CPU 环境使用 `tools/data/prepare_sites.py labels`；联网采样／产品查询使用已配置 GEE 的
环境（HPC 的 `/scr/u/xiaoz/HydroState/environments/gee-download/bin/python`），
模型检查使用 `/home/xiaoz/miniconda/envs/hydrostate/bin/python`。不要混用缺少 EE 的训练环境。

```bash
python tools/data/prepare_sites.py labels --root "$RAW_ROOT" \
  --pattern 'ismn/*/WATERLINE/Kerasia/*_sm_*' --source ismn \
  --start 2021-10-08 --end 2023-01-01 --as-of 2026-11-03 --output "$LABEL_OUTPUT"
python tools/data/prepare_sites.py pilot --label-record "$LABEL_RECORD" \
  --config configs/data/collection/site.py --output "$PILOT_ROOT" \
  --target-times 2021-12-03 2022-02-10 2022-03-14 2022-05-09 2022-10-24 2022-12-19
python tools/data/prepare_sites.py compare-sm --manifest "$PILOT_ROOT/samples.parquet" \
  --project cropland-477906 --output "$PILOT_ROOT/reports/product_consistency"
python tools/data/prepare_sites.py validate-model --manifest "$PILOT_ROOT/samples.parquet" \
  --config configs/hydrostate_unet.py --product-statistics "$PRODUCT_STATS" \
  --output "$PILOT_ROOT/reports/model_validation"
```

上述年月仅是这次小试范围，站点生产没有被限制为 2017–2024。

## 下一验收边界

多站点／多频率组的空间代表性与产品对照验收、跨来源 ET/FFP、站点分组与产品重合检查、
成功配额与 30 天间隔的正式分层随机抽样、联合读取与训练统计、SCR 发布。
小试数据仅为开发数据；没有固定且核验过的站点划分前，不进入正式验证／测试指标。
