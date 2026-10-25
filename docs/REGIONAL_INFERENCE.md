# 区域输入准备与当前模型推理

正式入口是 `tools/prepare_region.py` 和 `tools/predict_region.py`，使用 Drive 批量导出。
业务代码保留在 `hydrostate/data/regional/`，不归入 `legacy`。此流程只采集输入，
不下载产品标签，也不启动或改变训练任务。

## 输入定义与计算分工

复用 `retrieval/contract.py`、`specs.py`、`composites.py` 和 `alignment.py`。
五个连续 8 天槽以目标 PML 周期起点定位；`start/end` 筛选目标周期的**起始日期**，
左闭右开。年末目标周期按实际天数结束，输入槽仍严格沿用训练定义的五个 8 天槽。
左右上下文包含未来观测，属于回溯反演。区域配置的初始范围为 2017–2024 年。

GEE 完成逐景 QA、Landsat L1 TOA 转换、同 UTC 日期去重合成、有效日期等权均值，
以及有效日期数、平均/最早/最晚日期偏移。S1 在线性功率域平均后转回 dB。
每个空间块、每个发布分辨率组，把最多五个尚未缓存的槽打包成一次 Drive 导出。
数据与观测支持一起导出，不以零值判断缺测；确认无影像或全部无效可以是成功的空结果，
网络失败和不完整 GeoTIFF 不能发布为空样本。

HPC/本地负责元数据规划、任务监控、文件校验、训练一致的归一化和 Zarr 入库。
S2 的 10/20/60 m、Landsat 的 15/30 m 和 S1 的 10 m 组分别保存；这是产品发布
网格间距，不代表传感器实际观测分辨率提高。读取窗口时才对齐到 120×120 的 10 m 输入。

**S1 轨道不能因缓存复用而改变。** 每个窗口按其完整五槽候选影像，调用与训练相同的
`radar_orbit()`，选择同一相对轨道及升降轨方向。缓存身份包含轨道信息；邻近窗口或
日期只有选择相同时才共享对应 S1 合成。光学数据直接共享相同槽及空间块。

## 执行

使用项目 Python 3.12 环境，安装现有 `geo`、`gee` 依赖，提供有 GEE 和 Drive 权限的
用户凭据。默认使用 Earth Engine 持久凭据，也可用 `--drive-token /path/to/token.json`
指定 Drive 授权。凭据文件不能写入仓库。

```bash
export HYDROSTATE_REGION_ROOT=/path/to/inference/poyang_5x8_pilot

python tools/prepare_region.py plan \
  --config configs/inference/poyang_pilot.json --root "$HYDROSTATE_REGION_ROOT"
python tools/prepare_region.py prepare --root "$HYDROSTATE_REGION_ROOT"
python tools/prepare_region.py status --root "$HYDROSTATE_REGION_ROOT"
python tools/prepare_region.py audit --root "$HYDROSTATE_REGION_ROOT"

python tools/predict_region.py \
  --root "$HYDROSTATE_REGION_ROOT" \
  --model-config configs/hydrostate_mtlora_shared.py \
  --checkpoint /path/to/trained_checkpoint.pth \
  --pretrained /path/to/OlmoEarth-v1_2-Base \
  --statistics /path/to/training/target_stats.json \
  --batch-size 2 --workers 0 --device cuda
```

MTLoRA 使用 `configs/hydrostate_mtlora.py`，共享 LoRA 使用上面的 shared 配置；
配置必须与训练 checkpoint 对应，权重严格加载。预训练目录包含 `config.json` 和
`weights.pth`；统计量只能用对应训练集的文件，不能从区域影像重算。
`--dates 2020-07-11` 可只推理某个已准备的目标周期。推理只读取本地数据，不需要 GEE。

完整鄱阳湖使用 `configs/inference/poyang.json` 和新的输出根目录。
更换区域时配置 WGS84 `aoi.bbox`、`aoi.geojson`（Geometry 或 Feature 文件），
或 GEE FeatureCollection 与唯一 ID；提供覆盖该区域、以米为单位的投影。
不能直接用一个 UTM 投影处理全球或跨越很大的区域。

HPC IO 启动包装：

```bash
export HYDROSTATE_CODE_ROOT=/path/to/committed/source
export HYDROSTATE_PYTHON=/path/to/hydrostate/bin/python
# 冻结发布可以额外设置 HYDROSTATE_SOURCE_HASHES。
bash scripts/run_region_download_io.sh --root "$HYDROSTATE_REGION_ROOT"
```

代码同步仍走 Git；不要覆盖正在运行的冻结发布。采集使用 CPU，模型推理默认单 GPU。
本次试采不意味着已经启动全域采集。

## 文件组织与断点

```text
region_root/
  plan.json, aoi.geojson       不可变输入定义、WGS84 AOI、网格、归一化和目标周期
  drive.json                  可调整的运行并发与容量预算
  windows.parquet             一次生成的模型窗口索引
  catalog/                    按传感器及月份缓存的影像元数据
  targets/                    每个日期的窗口轨道选择和所需共享产品清单
  exports.sqlite              Drive 任务身份、文件校验信息与恢复状态
  inputs/xx/<identity>.zarr/   一个时间槽×空间块×波段组×轨道的共享缓存
    values, valid, quality
    _SUCCESS.json             校验、归一化和全部数组写入完成后发布
  manifests/<date>.json        所需产品全部到齐后发布；模型只读取这些日期
  staging/drive/              尚未确认入库的导出文件
  products/<identity>.zarr/   对应输入和 checkpoint 的区域产品
```

共享产品采用独立 Zarr，避免多个线程竞争同一个全区域数组的元数据。
每个模型样本仅保存窗口和日期索引，不复制五槽数据。缓存与旧四槽目录严格分开。

原命令重跑会接回已有远端任务、复用成功产品，已完成日期跳过规划。
明确失败的远端任务保留错误；排除原因后通过 `--retry-failed` 重试。
修改 Drive 并发或容量后，重新执行带相同科学配置的 `plan` 更新运行设置；
更改 AOI、日期范围、块大小或其他输入定义须使用新根目录。当前尚不支持原地扩展计划日期。

队列默认最多 2 个在途逻辑导出、2 个下载入库工作线程，加 1 个有界规划线程；
不为每个任务重新创建线程池。规划和下载不阻塞主线程的任务监控。
只有完整入库后才删除本流程记录的 Drive 导出文件及对应 staging 文件，
不清空用户回收站。容量不足会在 `capacity.json` 指明 Google Drive、HPC scratch
或在途预算哪一项不足，保留任务状态。

## 区域输出与验证含义

输出七个变量、派生 `et_total`、各变量有效掩膜、`time_bounds`、`x/y/time` 坐标和
`date_complete`。只有 `date_complete == 1` 的日期可用。Zarr 带 CRS、30 m 仿射变换、
checkpoint/统计量哈希和模型配置。可用 `xarray.open_zarr(path, mask_and_scale=False)`
读取，保留有效性数组中的 0；时间坐标为目标周期中点，边界才是实际时间支持。

模型窗口固定 1.2 km，默认保留中心 32×32 个输出像元、480 m 步长，用余弦权重融合。
覆盖比例共用融合权重，因此仍闭合。条件 SM 另按预测 LF 加权融合；不对 SM 乘 LF 导出，
并按模型陆地门槛屏蔽无效域。三类 ET 也按对应表面比例加权融合各自的**条件率**，
再由融合后的 WF/IF/LF 合成总 ET，保持重叠窗口的面积贡献一致。
某表面比例严格为零时其分支率不定义，对总量贡献为零；这不涉及产品缺失标签补零。

融合累加采用磁盘映射，按块写出，避免一次把多年全域输出装入内存。
空间重叠平滑与覆盖闭合是实现检查，不是精度证据；小比例表面的条件 ET 也没有因此
获得独立验证。当前没有自动关联区域与训练样本的时空重叠标记，地图不能直接用于
声称独立空间泛化；独立验证需另行核查训练重叠。

验收记录见 [2026-10-25 区域端到端验证](validation/REGIONAL_20261025.md)。
