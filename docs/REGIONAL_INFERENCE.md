# 区域输入准备与当前模型推理

正式入口是 `tools/prepare_region.py` 和 `tools/predict_region.py`。输入默认通过
`computePixels` 直接获取，`--backend drive` 显式选择 Drive 批量导出。
业务代码保留在 `hydrostate/data/regional/`，不归入 `legacy`。此流程只采集输入，
不下载产品标签，也不启动或改变训练任务。

## 输入定义与计算分工

复用 `retrieval/contract.py`、`specs.py`、`composites.py` 和 `alignment.py`。
五个连续 8 天槽以目标 PML 周期起点定位；`start/end` 筛选目标周期的**起始日期**，
左闭右开。年末目标周期按实际天数结束，输入槽仍严格沿用训练定义的五个 8 天槽。
左右上下文包含未来观测，属于回溯反演。区域配置的初始范围为 2017–2024 年。

GEE 完成逐景 QA、Landsat L1 TOA 转换、同 UTC 日期去重合成、有效日期等权均值，
以及有效日期数、平均/最早/最晚日期偏移。S1 在线性功率域平均后转回 dB。
每个空间块、每个发布分辨率组，把尚未缓存的槽按字节预算打包成请求。
直接像素通道默认每请求不超过 32 MiB：完整 9.6 km 块的 10 m 组单槽获取，
较粗分辨率组可合并多槽。Drive 通道每任务最多五槽，使用自己的导出容量预算。
两条通道共用同一个合成表达式、产品身份和入库逻辑。
数据与观测支持一起返回，不以零值判断缺测；目录查询成功且确认无源景的请求直接在
本地生成全缺测产品。有源景但 QA 后全部无效仍需实际计算，网络失败不能发布为空样本。

HPC/本地负责元数据规划、任务监控、文件校验、训练一致的归一化和 Zarr 入库。
S2 的 10/20/60 m、Landsat 的 15/30 m 和 S1 的 10 m 组分别保存；这是产品发布
网格间距，不代表传感器实际观测分辨率提高。读取窗口时才对齐到 120×120 的 10 m 输入。

**S1 轨道不能因缓存复用而改变。** 每个窗口按其完整五槽候选影像，调用与训练相同的
`radar_orbit()`，选择同一相对轨道及升降轨方向。缓存身份包含轨道信息；邻近窗口或
日期只有选择相同时才共享对应 S1 合成。光学数据直接共享相同槽及空间块。

## 执行

使用项目 Python 3.12 环境，安装现有 `geo`、`gee` 依赖，使用 Earth Engine 持久凭据。
只有 Drive 通道需要额外的 Drive 权限；可用 `--backend drive --drive-token /path/to/token.json`
指定 Drive 授权。凭据文件不能写入仓库。

```bash
export HYDROSTATE_REGION_ROOT=/path/to/inference/poyang_5x8_pilot

python tools/prepare_region.py plan \
  --config configs/inference/poyang_pilot.json --root "$HYDROSTATE_REGION_ROOT"
python tools/prepare_region.py prepare --backend pixels --root "$HYDROSTATE_REGION_ROOT"
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
当前边界采用用户指定的 GEE 资产
[`users/xiaozhen6666666/ChinaWater/Poyang_Lake_region2`](https://code.earthengine.google.com/?asset=users/xiaozhen6666666/ChinaWater/Poyang_Lake_region2)，
已下载为 `configs/inference/regions/poyang_lake_region2.geojson` 并随 Git 固定版本，
不额外添加缓冲。它约覆盖 115.7833–116.7500°E、28.3667–29.7500°N，
UTM 50N 平面面积约 14,423 km²。旧 HydroLakes 外扩 10 km 计划仍保留在其原数据根目录；
更换边界必须新建数据根目录，不能复用旧计划身份或覆盖旧成果。
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
运行是否启动、已完成多少日期须查看对应运行回执、进程和日志。

HPC 当前鄱阳湖全量输入的存储根目录为：

```bash
export HYDROSTATE_REGION_ROOT=/fossfs/xiaozhen/HydroState/inference/poyang_retrieval_5x8_201704_202412_f9fce12
```

采集和模型推理都直接传入这个 `--root`；旧 `/scr/u/xiaoz/HydroState/inference`
不保留路径别名。目录名末尾的提交号记录最初建库来源，当前执行代码以
`submission.json` 的 commit 和冻结源码路径为准。
迁移校验及续跑记录见 [FossFS 迁移验收](validation/REGIONAL_FOSSFS_MIGRATION_20261025.md)。

从正在运行的当前五槽 Drive 采集切换时，先停止其本地调度进程，再完成已提交任务：

```bash
python tools/prepare_region.py prepare --backend drive --drain --root "$HYDROSTATE_REGION_ROOT"
python tools/prepare_region.py plan --config configs/inference/poyang.json \
  --root "$HYDROSTATE_REGION_ROOT"
bash scripts/run_region_download_io.sh --backend pixels --root "$HYDROSTATE_REGION_ROOT"
```

`--drain` 只处理已经记录的 Drive 任务，不生成新区域请求。直接通道在持有同一准备锁后
检查 Drive 已清空，未完成时拒绝启动。相同科学配置重新 `plan` 只更新运行设置，
不改写 `plan.json` 或已有产品，切换无需重新下载成功结果。

## 文件组织与断点

```text
region_root/
  plan.json, aoi.geojson       不可变输入定义、WGS84 AOI、网格、归一化和目标周期
  drive.json, pixels.json      各传输通道可调整的运行并发与容量预算
  windows.parquet             一次生成的模型窗口索引
  catalog/                    按传感器及月份缓存的影像元数据
  targets/                    每个日期的窗口轨道选择和所需共享产品清单
  exports.sqlite              Drive 任务身份、文件校验信息与恢复状态
  pixels_progress.json         直接通道本次运行计数、有效像素量和最近状态
  pixels_failure.json          最近失败请求及原因（历史证据，不代表当前仍失败）
  logs/pixels_receipts.jsonl   直接请求入库回执，包含耗时、产品身份和空源标记
  inputs/xx/<identity>.zarr/   一个时间槽×空间块×波段组×轨道的共享缓存
    values, valid, quality
    _SUCCESS.json             校验、归一化和全部数组写入完成后发布
  manifests/<date>.json        所需产品全部到齐后发布；模型只读取这些日期
  staging/drive/              尚未确认入库的导出文件
  products/<identity>.zarr/   对应输入和 checkpoint 的区域产品
```

共享产品采用独立 Zarr，避免多个线程竞争同一个全区域数组的元数据。
每个模型样本仅保存窗口和日期索引，不复制五槽数据。缓存与旧四槽目录严格分开。

直接通道重跑以原子成功标记恢复，只请求缺失产品；不会重写已完成数组。
Drive 重跑会接回已有远端任务。已完成日期均跳过规划。
直接请求仅对暂时性服务/连接错误有限重试（默认最多 3 次）；永久错误和无效数组直接失败。
Drive 明确失败的远端任务保留错误，排除原因后通过 `--backend drive --retry-failed` 重试。
修改任一通道的并发或容量后，重新执行带相同科学配置的 `plan` 更新运行设置；
更改 AOI、日期范围、块大小或其他输入定义须使用新根目录。当前尚不支持原地扩展计划日期。

直接通道默认 4 个获取/入库线程，加 1 个有界规划线程，使用同一线程池。
每请求最多 32 MiB、2 CPU、0 GPU；不为每个模型窗口或请求重新建池。
若单槽超出配置预算，明确报错，使用 Drive 或新建较小空间块的计划，不静默更改数据定义。
Drive 默认最多 2 个在途逻辑导出、2 个下载入库线程，加 1 个有界规划线程。
两条通道都检查输出目录所在文件系统的余量，并保留默认全域 20 GiB 预留。
只有完整入库后才删除本流程记录的 Drive 导出文件及对应 staging 文件，
不清空用户回收站。容量不足会在 `capacity.json` 指明 Google Drive、输出文件系统
或在途预算哪一项不足，保留任务状态；`hpc_scratch` 是现有诊断字段名，
空间检查实际作用于传入的根目录，目录迁入 FossFS 后检查的也是 FossFS。

## 区域输出与验证含义

### 单卡 HPC 推理

FossFS 输入必须提交到 `c_foss_gpu`。`scripts/slurm/predict_region.sh` 固定单 GPU、
6 CPU、64 GiB 内存和最长 7 天；先在登录节点确认当前 account/QoS 限制。
代码经 Git 同步后冻结发布，设置以下运行环境，再提交一次并立即记录 job ID：

```bash
export HYDROSTATE_CODE_ROOT=/path/to/frozen/release
export HYDROSTATE_CODE_COMMIT='full-git-commit'
export HYDROSTATE_SOURCE_HASHES=/path/to/source.sha256
export HYDROSTATE_REGION_MODEL_CONFIG=/path/to/training/expanded_config.py
export HYDROSTATE_CHECKPOINT=/path/to/training/best_checkpoint.pth
export HYDROSTATE_MODEL_PATH=/path/to/OlmoEarth-v1_2-Base
export HYDROSTATE_TARGET_STATS=/path/to/training/provenance/target_stats.json
export HYDROSTATE_RUN_DIR="$HYDROSTATE_REGION_ROOT/runs/<unique-run-name>"
mkdir -p "$HYDROSTATE_RUN_DIR"
sbatch --parsable --output="$HYDROSTATE_RUN_DIR/slurm-%j.out" \
  --error="$HYDROSTATE_RUN_DIR/slurm-%j.err" \
  "$HYDROSTATE_CODE_ROOT/scripts/slurm/predict_region.sh" \
  --pilot-dates 2017-04-07 2020-07-11 2022-11-01
```

启动检查包括全部目标日期的清单身份、三个代表日期各八个分散窗口的实际读取、
checkpoint 严格加载，以及真实 GPU 上 batch 2/4/8/16 的前向测试。
选取保留至少 15% 显存余量、达到实测最高吞吐 95% 的最小 batch。
该测试复用已读取输入，包含预处理和传回 CPU 的耗时，**不包含磁盘读取和拼接**；
正式运行的每分钟 `windows_per_second` 才是流水线吞吐，首个完整日期后再估算总工期。
这不是遍历全部输入数组的审计；未抽中的产品在正式读取时检查成功标记并读取。

`--pilot-dates` 优先完成并画出这些日期，之后继续所有日期；它不等同于 `--dates`
只选择部分日期。预览包含七个变量及总 ET，英文 PNG。SM/ET 表面比例低于 5%
默认只在预览中隐藏（`--display-surface-min`），原始条件率和总 ET 不受显示门槛影响。

`runs/<name>/inference.json` 保存模型、输入、commit、状态和最终产品路径，
`startup.json` 保存读取及显存/吞吐测试。`products/<identity>_progress.json` 每分钟更新；
`products/<identity>_<date>.json` 记录单日期耗时、有效面积、物理范围和覆盖闭合误差。
只有全部数组写入且检查通过后才发布该日期的 `date_complete`，BF16 覆盖闭合允许
0.01 的数值误差并记录实际误差。预览位于 `products/<identity>_previews/`。
续跑使用相同冻结代码、输入、配置、统计量和 checkpoint，完成日期直接跳过；
中断日期从头重算。重提之前先核对队列，禁止两个作业同时写同一个产品。

### 产品定义

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

验收记录见 [2026-10-25 区域端到端验证](validation/REGIONAL_20261025.md)及
[直接像素与 Drive 对照](validation/REGIONAL_PIXELS_BENCHMARK_20261025.md)。
当前入口的实现与试采见 [直接通道切换验证](validation/REGIONAL_PIXELS_SWITCH_20261025.md)。

## 推理性能诊断

`tools/benchmarks/region_inference.py` 是薄入口，计时代码在
`hydrostate/inference/regional_benchmark.py`。使用与正式推理相同的 checkpoint、统计量、
输入和 BF16 设置；诊断只读取输入，并在独立输出目录写报告、算子 trace 和临时融合累加器，
不发布或修改区域产品。GPU 测试提交 `c_foss_gpu` 的 debug QoS，单卡、适量 CPU，
按 debug 时间上限缩小测试窗口，不能把正式全量任务当作测试。

```bash
python tools/benchmarks/region_inference.py \
  --root "$HYDROSTATE_REGION_ROOT" --model-config "$HYDROSTATE_REGION_MODEL_CONFIG" \
  --checkpoint "$HYDROSTATE_CHECKPOINT" --pretrained "$HYDROSTATE_MODEL_PATH" \
  --statistics "$HYDROSTATE_TARGET_STATS" --output /path/to/new-diagnostic-directory \
  --dates 2017-04-07 2020-07-11 --start-index 27000 --windows 32 \
  --batches 2 4 8 16 20 24 --workers 0 2 4
```

报告区分原生数组读取、对齐、输入预处理/设备传输、模型预测、回传 CPU 和重叠融合。
缓存 batch 测试与包含读取的流水线测试分别报告吞吐；流水线另记录暴露在 GPU 计算之外的
数据等待时间，并用算子 profiler 定位模型内部耗时。重复读取仅指应用层重复，
不声称已清空操作系统缓存；短流水线测试也不包含整日期最终压缩写出。
