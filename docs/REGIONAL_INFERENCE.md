# 鄱阳湖三变量区域推理

使用任务 4251319、epoch 32 的三变量多尺度权重。生产时间为 2017-04-01
至 2025-12-31，空间范围为 HydroLAKES v1 湖泊 151（Poyang）加 10 km 固定缓冲。
这是一种可复现的湖区及周边范围定义，不是独立湿地名录或行政区边界。
配置位于 `configs/inference/poyang_3vars.json`。

## 网格、时间与数据含义

采用 EPSG:32650（UTM 50N）：鄱阳湖约 116°E，处在 114°E–120°E 的第 50 带。
米制坐标便于统一 10 m 输入、30 m 输出、滑窗及缓冲区。经纬度可以由保存的
CRS 和 x/y 坐标转换，导出时也可以重投影。

时间轴保持 PML 每年从 1 月 1 日开始的 8 日周期，参考日期取区间中点。
以参考日期筛选请求范围，共 403 期：2017-04-03 至 2025-12-29。
首期 ET 的完整支持区间是 `[2017-03-30, 2017-04-07)`；没有将部分区间误称为
完整 8 日均值。最后一期和闰年尾期均按实际天数记录。WF/SM 是参考日期附近的
重建，ET 是所属区间的平均日速率（mm/day）。不要将 WF/SM 自动解释为 8 日均值，
或将 ET 当作瞬时值。月/年累计 ET 必须按各区间实际重叠天数加权。

旧权重没有统一查询时间模块。每期移动过去 63 天的四个时间槽；采用原来的
S1、S2 SR 和 Landsat L1 波段、QA 掩膜、单景选择、同轨 S1 及归一化。
每景每窗口至少 50% 有效，至少两个有效时间槽，最近观测不超过 3 天。
没有满足条件的窗口保留缺测。最初几期的上下文早于 2017 年 4 月，缺失传感器
由有效性掩膜表示；以真实 GEE 影像查询结果确认覆盖，不以目录起止日期推定。

每个输入窗口为 120×120×10 m，输出 40×40×30 m。保留中心 32×32，
窗口步长 480 m，中心区域采用余弦权重融合，另报告重叠窗口预测的离散程度。
边缘 120 m 仅作上下文，避免将解码器外缘直接写入产品。不能直接增大输入
窗口代替拼接，因为注意力和 GroupNorm 的上下文会随之改变。拼接连续性须在
试运行中检查；中心裁剪并不保证消除所有接缝。

WF 为 0–1 比例；SM 为 m³/m³，由预测 WF 按训练配置排除近乎纯水体。
ET 输出的湖面部分单独标记未验证，不能自动当作湖面蒸发。30 m 网格不表示
SM/ET 已获得独立验证的 30 m 有效分辨率。

## 执行

有 GEE 权限的机器负责 `init`/`prepare`，GPU 机器只读取缓存，不需要 GEE 凭据。
使用已有 `hydrostate` Python 环境及 GEE 项目。先生成和核对试验区：

```bash
python tools/predict_region.py init \
  --root outputs/poyang_201704_202512_pilot --pilot \
  --training-manifest /path/to/original/samples.parquet
python tools/predict_region.py prepare \
  --root outputs/poyang_201704_202512_pilot --workers 4
python tools/predict_region.py status --root outputs/poyang_201704_202512_pilot
```

`--dates 2017-04-03` 可以只准备或预测指定参考日期。`prepare` 不依赖目标标签，
按影像及空间块下载，缓存可供相邻窗口和日期复用。成功窗口记录在 SQLite 中，
完整日期另有 Parquet 清单；网络失败后重跑同一命令即可恢复。

也可以在 HPC 上直接下载；这只需要 CPU，无需占用 GPU。本次 HPC 的计算节点
无法联网，使用官方文件传输节点 `hpc2021-io1.hku.hk` 或 `hpc2021-io2.hku.hk`
后台运行，不提交 Slurm 下载任务，也不在主节点长期运行下载。GEE 用户凭据放在
`~/.config/earthengine/credentials`，目录权限为 `700`、文件权限为 `600`，
通过 SSH 单独传输，不能提交到 Git。`client_secret.json` 本身不等于已完成的
用户授权，直接像素下载实际使用 Earth Engine 凭据。

完整区域现在使用 Drive 批量导出通道；小样本继续使用像素下载。
区域流程先通过单字节掩膜选片，再仅导出入选影像所需空间块。`selections/`
表示选片完成，只有 `manifests/` 中的日期才表示完整输入就绪；推理读取后者。
代码结构、配置和恢复流程见 [GEE_DOWNLOAD.md](GEE_DOWNLOAD.md)。
使用已安装 GEE 和本项目依赖的 Python 3.12，在 IO 节点运行。先设置冻结源码目录、
对应源码校验清单和解释器路径：

```bash
export HYDROSTATE_CODE_ROOT=/path/to/frozen/source
export HYDROSTATE_SOURCE_HASHES=/path/to/source.sha256
export HYDROSTATE_GEE_PYTHON=/scr/u/xiaoz/HydroState/environments/gee-download/bin/python
nohup bash "$HYDROSTATE_CODE_ROOT/scripts/run_region_download_io.sh" \
  --root /scr/u/xiaoz/HydroState/inference/poyang_201704_202512 \
  --workers 4 --min-free-gib 20 > /scr/u/xiaoz/HydroState/logs/poyang_inputs.log 2>&1 < /dev/null &
```

省略 `--dates` 会依次处理全部 403 期，已有结果会跳过。CLI 默认保留 20 GiB
空间；达到阈值或请求持续失败时退出并保留断点，不会自动删除缓存。导出队列受
并发数和总字节数限制；Drive 文件校验入库后清理。用相同命令可继续，重启前先
确认旧进程已退出，避免重复启动。此次正式后台进程使用冻结源码，并将 PID、
主机、命令、环境、源码版本和日志路径记录在区域目录的 `submissions/download_*.json`。
IO 启动器将 CPU 限制为最多两个，并降低调度优先级；下载与选片仍支持四路网络并发。

只有计算节点可联网时才使用 CPU 作业脚本 `scripts/slurm/31_prepare_region.sh`，
并设置 `HYDROSTATE_CODE_ROOT`、`HYDROSTATE_SOURCE_HASHES` 和
`HYDROSTATE_GEE_PYTHON`，分别指向冻结源码、源码校验清单和可用解释器。

通过 rsync 将 `plan.json`、`aoi.geojson`、`windows.parquet`、`catalogs/`、
`manifests/` 和 `input_cache/` 同步至 HPC。缓存和产品都是普通目录式 Zarr，
不用 ZIP 或自定义分片压缩包。数据可放在：

```text
/scr/u/xiaoz/HydroState/inference/poyang_201704_202512_pilot/
```

代码 commit/push 后，在 `/home/xiaoz/HydroState` fast-forward 拉取，再提交：

```bash
bash scripts/slurm/29_submit_region.sh \
  /scr/u/xiaoz/HydroState/inference/poyang_201704_202512_pilot \
  --dates 2017-04-03
```

提交器冻结源码快照，验证权重、训练清单和代表性缓存读取，保存唯一提交记录。
相同版本/输入/日期的重复提交会失败；先检查记录中的 job ID，再处理失败重试。
预测默认 batch=8，单 GPU；本机小显存测试可直接用
`tools/predict_region.py predict --batch-size 1 --workers 0`，并通过
`--checkpoint`、`--pretrained`、`--statistics` 指向本机文件。

`predict --drop-s2` 生成单独的 `poyang_3vars_without_s2.zarr`，保持相同窗口及
其余影像，供缺失 S2 消融比较。比较必须使用两次预测共有的有效支持；只有 S2
可用的窗口不能参与消融。预测差异本身不是精度损失，需要参考观测评价。

## 输出与重叠标记

主产品 `products/poyang_3vars.zarr` 包含 `water`、`soil_moisture`、`et`，维度
为 `(time,y,x)`，chunk 为 `(1,512,512)`，使用普通 Zarr v2 与 Zstd 压缩。
包含 CRS、单位、时间边界、权重 SHA256、预处理配置和训练清单身份。
`date_complete` 写在该期所有数据之后；读取时仅使用已完成日期。

`quality` 是逐像元位掩膜（可同时具有多个标记）：

| 位值 | 含义 |
|---|---|
| 1 | WF/ET 有有效输入支持 |
| 2 | SM 有效 |
| 4 | 覆盖此像元的窗口中存在无 S2 输入的窗口 |
| 8 | 输入窗口空间相交，且与训练样本的 63 日输入上下文相交 |
| 16 | 输入窗口空间相交，且同一个目标 8 日区间内有训练样本 |
| 32 | 此像元落在训练样本的空间足迹内 |
| 64 | 预测 WF≥0.99，湖面 ET 未单独验证 |

重叠只标记，不删除、不阻止推理。多个窗口覆盖一个像元时，描述性标记按位或合并。
位 8/16 是保守的窗口级时间重叠，并非证明
使用了完全相同的卫星影像；清单保留匹配的 sample_id，便于进一步核查。
`observation_time(time,window_y,window_x,slot,sensor)` 保存每景 UTC 时间；
0 表示该传感器/时间槽缺失。逐期报告记录有效覆盖、范围和运行耗时。

## 扩展到完整区域

不带 `--pilot` 初始化独立完整区域目录，先按日期或按月准备、预测和验收，避免
一次下载十年的输入撑满 scratch。输入缓存应按完成日期滚动管理；已确认不再
用于后续 63 日上下文、消融或重跑的影像块才可清理。当前 HPC 约 200 GB 空余，
正式生产前需要依据试运行实测缓存量安排批次，不能假设全部 403 期输入都能
同时保留。若按全部日期下载，必须保留上述空间阈值，并检查作业是否已暂停。

正式扩大前检查：区域边界、输入归一化与掩膜、缺失 S2 时的表现、湖岸与洲滩
接缝、各期覆盖率、SM/ET 的监督尺度以及独立参考或产品一致性。训练重叠不会
作为剔除条件，报告清楚区分重叠和未重叠部分。S2 消融与拼接检查通过后再扩展。

可以用 `python tools/validate_region.py --root ROOT --date 2017-04-03` 检查少量
内部窗口的教师产品一致性：WF 按 30 m 聚合，ET 使用完整 PML 原生像元面积权重，
SM 保留训练时的陆地加权窗口近似。这不构成独立验证；湖面附近 SM 参考可能完全
缺失，少量 ET 参考也不足以说明区域精度。`--offline` 复用下载好的参考。

首轮无重叠中心拼接发现 SM/ET 存在窗口跳变，故采用上述 480 m 步长融合。
重叠窗口差异仍会保存在逐期报告，不能用图面变平滑证明精度提高。

2026-10-16 的首期试验（2017-04-03，约 92 km²）中，中心直拼的 SM/ET 边界
平均梯度分别约为内部梯度的 5.6/7.2 倍；重叠融合后约为 0.9/1.0 倍。
这是拼接诊断，不是精度指标。同一批 9,332 个有效 WF 参考像元上，屏蔽 S2
使与 Dynamic World 的 RMSE 从约 0.017 增加到 0.318。当前权重对 S2 缺失
明显敏感，早期年份不能按相同可信度解读。SM 没有合格参考、ET 仅有两个
原生像元，不能据此报告区域 SM/ET 精度。先扩大试验再启动全域长序列生产。
