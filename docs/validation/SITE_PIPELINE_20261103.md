# 站点流程首轮实施（2026-11-03）

状态：实施中。已授权按[完整方案](../SITE_SAMPLE_PLAN.md)推进；本记录区分实现、
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

## 下一验收边界

真实小批 SM Zarr、SMAP 同日期对照、跨来源 ET/FFP、站点分组与产品重合检查、
成功配额与 30 天间隔的正式分层随机抽样、联合读取与训练统计、SCR 发布。
小试数据仅为开发数据；没有固定且核验过的站点划分前，不进入正式验证／测试指标。
