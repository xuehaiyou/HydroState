# GEE 数据获取：样本像素通道与区域 Drive 通道

两条通道共享产品、QA、时间槽、波段顺序和归一化定义。小样本默认使用
`computePixels`；大范围连续推理按影像和空间矩形批量导出至 Google Drive，
下载并验证后写入普通目录式 Zarr，再删除该次导出的临时文件。

## 代码位置与职责

```text
hydrostate/data/gee/
  products.py             产品、波段、QA、影像表达式和归一化
  contracts.py            网格、稳定请求身份、锁与原子元数据写入
  client.py               GEE 初始化与有限重试
  catalog.py              可复用的影像元数据目录
  selection.py            原有窗口观测选择规则
  region_plan.py          入选窗口的空间块并集、缺失块与有大小上限的区域矩形
  state.py                exports.sqlite 持久化导出状态
  backends/
    pixels.py             小请求 computePixels；区域单字节掩膜请求与并发去重
    drive.py              区域导出、Range 续传、校验、入库和精确清理
  workflows/
    samples.py            样本规划与获取；保留旧 Drive 样本实现
    region.py             区域准备、缓存复用和日期清单
    drive_queue.py        有并发数和字节数上限的导出队列
hydrostate/storage/scene_cache.py  离线 Zarr 读写，无网络请求
hydrostate/storage/quality_cache.py 离线掩膜 Zarr、已有输入复用和一致性验证
```

入口是 `tools/data/prepare_gee_samples.py` 和
`tools/data/prepare_region_inputs.py`；Drive 调度参数在
`configs/data/gee_drive.json`。`hydrostate/data/gee_sampling.py`、
`hydrostate/inference/gee.py` 和 `tools/predict_region.py prepare` 保留兼容入口。
推理代码直接读取离线 SceneCache，不需要 GEE/Drive 授权。
新增产品修改共享产品定义；修改传输方式只改 backend；调度策略在 workflow 中修改。

## 样本通道

```bash
python tools/data/prepare_gee_samples.py \
  --project cropland-477906 --output /path/to/samples \
  --samples 10000 --backend pixels --concurrency 2
```

`scripts/run_gee_sampling.sh` 默认走 pixels，可设置 `GEE_BACKEND=drive`
并提供 `DRIVE_TOKEN` 重用旧流程。样本规划不变，两种传输都走相同入库校验。
新增标签的 `gee_labels.py` 也使用共享像素客户端；产品选择与标签质量规则不变。
样本像素响应通过内存 TIFF 交给原入库函数，不积累临时 TIFF；样本完成清单逐条发布。

## 区域通道

先用 `tools/predict_region.py init` 建立区域 plan，已有 plan 不需重新建立。

```bash
python -u tools/data/prepare_region_inputs.py \
  --root /scr/u/xiaoz/HydroState/inference/poyang_201704_202512 \
  --download-config configs/data/gee_drive.json \
  --workers 4 --min-free-gib 20
```

此入口默认 Drive。可用 `--dates 2017-04-03` 只准备一期；省略则依次准备全部期次。
旧入口仍默认 pixels，使用区域导出时显式加 `--backend drive`。
本项目 HPC 的 CPU 计算节点无法联网。下载使用官方文件传输节点
`hpc2021-io1.hku.hk` 或 `hpc2021-io2.hku.hk`，不在主节点 `hpc2021.hku.hk`
长期运行下载进程。节点用途见 [HKU 官方说明](https://hpc.hku.hk/hpc/hpc2021/userguide/)。
IO 节点不经过 Slurm 排队；使用 `scripts/run_region_download_io.sh` 检查主机和
冻结源码校验和，并将进程限制在最多两个 CPU，设置 nice=10 和计算库单线程。
四个选片线程主要用于网络并发。启动记录必须包含 hostname、PID、commit 和日志，
后续通过同一 IO 节点查询进程；共享 `prepare.lock` 防止跨节点重复启动。

默认复用 `~/.config/earthengine/credentials`；授权需要包括 Drive 权限，
对应 quota project 必须启用 Drive API。也支持 `--drive-token /private/path/token.json`
以及 `--drive-quota-project`。密钥、token、数据和运行记录不进入 Git。

每个区域目录有唯一命名的 Drive 临时文件夹，程序拒绝同名文件夹歧义。
配置默认最多两个导出、两个文件下载线程，单导出原始数据最多 0.5 GiB，
待处理原始数据总额最多 1 GiB，另外保留 0.5 GiB Drive 空间。
这适配当前账号较小的剩余空间；扩容后可调整这些参数。
每个 TIFF 部分最多 3840×3840 像素，对齐 256 像素 GEE shard 和 480 像素缓存块。
这只是传输期间的 GeoTIFF 切块，最终 Zarr 不使用 ZIP。

区域准备分为两阶段：

1. 按需读取候选影像的单字节有效性掩膜，先完成窗口选片。已有完整输入块直接复用；
   缺失掩膜使用 `computePixels`，默认 1920×1920 像素一块，保持原 10 m 网格。
   相同掩膜请求去重，不同块可并行；网络请求不持有写锁。
2. 仅对通过筛选的窗口，合并入选影像所需的 480×480 空间块，包括输入 halo。
   跳过已完成块，将空间需求按大小上限递归拆成矩形；额外填充面积不超过需求的 50%，
   由 `configs/data/gee_drive.json` 的 `max_padding_fraction=0.5` 控制。
   单任务仍最多 0.5 GiB，填充像素也计入大小上限；只把必需块写入缓存。
   这是传输区域合并参数，不改变模型窗口、QA 或入选观测。
   这些完整波段通过 Drive 导出、下载、校验、入库和清理。

选片仍在本地完成：四个时间槽、按时间与资产 ID 倒序、逐窗口锁定 S1 轨道、
有效像素比例、最新观测时效及缺失传感器规则不变；不使用整景云量代替窗口 QA，
不制作时间合成影像。掩膜是 QA 与**所有输入波段有效范围的交集**，与完整导出的
`valid` 完全相同；仍在原 30 m 像素中心计算比例，不先将掩膜粗化。
完整输入下载后再次比较对应掩膜，全部所需块就绪才发布 `manifests/`。

有效性单独导出为 valid 波段，
空像素显式为零；关闭 skipEmptyTiles，从而能严格区分空区域与尚未下载的 TIFF 部分。
矩形包含少量填充像素，但不会为未入选影像新建完整波段导出。实际端到端提速
取决于入选比例、已有缓存及 GEE 排队；`reports/` 记录每期需求量和两阶段耗时。
报告的 `packing` 字段记录该次任务规划使用的大小与填充上限。未指定
`max_padding_fraction` 的旧配置仍按 25% 处理；重启时先接收原有在途任务，再按新参数
规划剩余缺失块。已有缓存不需要重建，稳定的导出身份也不因配置名称改变。

## 恢复与数据安全

鄱阳湖三变量推理的 `configs/inference/poyang_3vars.json` 使用 `latest_days=4`：
最近一次入选观测到目标日期最多相隔 4 天；窗口有效像元比例仍为 50%，至少两个
时间槽有观测。样本采集的默认 3 天规则保持不变。

已有区域的参数保存在 `plan.json`，修改源配置不会自动改变运行断点。需要放宽
观测时效时，先停止该区域的准备进程，执行离线修订，再用原命令恢复下载：

```bash
python tools/data/update_region_recency.py --root /path/to/region --latest-days 4
```

工具持有准备、预测和下载锁，在 `revisions/` 备份原计划、筛选数据库与受影响的清单，
按已保存的观测年龄重算接受状态。只有接受状态变化的日期才撤销就绪清单，补齐缺失
完整波段后重新发布；已有影像、掩膜和在途 GEE 导出均保留。已有预测产品时工具拒绝
原地修订，避免把旧预测标成新规则。若修订中断，正常准备入口会阻止读取混合状态，
重新运行同一修订命令即可接续。该工具仅支持放宽时效，不更改选片或 QA 规则。

```text
区域根目录/
  plan.json / windows.parquet / catalogs/  原区域计划与目录
  input_cache/<sensor>/<scene>.zarr/       已归一化输入，跨日期复用
  quality_cache/<sensor>/<scene>.zarr/     单字节有效性掩膜，不代表完整输入已就绪
  selections/<date>.parquet                选片完成，完整波段可能尚未下载
  manifests/<date>.parquet                 完整输入就绪，可用于推理
  selection_contract.json                  防止变更计划后误用选片断点
  reports/preparation_<date>.json          所需块、导出量、掩膜请求数与阶段耗时
  preparation.sqlite                      可恢复的逐窗口结果
  exports.sqlite                          导出身份、task ID、文件、提交状态
  staging/drive/<export_id>/               临时 TIFF 和 .part 续传文件
  logs/ / submissions/                    启动日志与版本记录
```

- 区域像素和 Drive 工作流共用 `prepare.lock`，不会同时写入。
- 已有兼容缓存的 done 块直接复用；缓存网格、归一化和块尺寸必须一致。
- 切换到掩膜选片时，先接收 exports.sqlite 中已提交的旧导出，再开始新需求规划。
  preparation.sqlite 中原规则生成的窗口结果仍可复用；不删除现有数据。
- 先记录 GEE request ID 再提交；中断后检查原任务并复用其 ID，避免重复提交。
- 下载以文件 ID、长度和 MD5 固定对象身份；HTTP Range 续传，完成后校验 MD5。
- TIFF 验证 CRS、分辨率、像素原点、波段顺序、全部部分的覆盖和非重叠性。
- 写入 Zarr 后读回校验并 fsync 对应数据块，再发布 done 标记。
- SQLite 记录 committed 后才按文件 ID 删除该次导出；清理中断可重跑。
  不清空 Drive，不清空回收站，也不删除其他文件。
- 下载失败、远端任务失败或空间不足会保留状态和已完成缓存。使用相同命令恢复；
  远端 FAILED/CANCELLED 任务保留诊断，不无限重提或覆盖其文件。

`--min-free-gib 20` 是本地保护线，不代表全部 2017–2025 输入只需当前剩余空间。
长期完整影像缓存可能需要扩容；该流程不会自动删除已下载的历史影像。

真实传输一致性验证使用独立输出目录，每个传感器选一个已有有效缓存块：

```bash
python tools/data/validate_region_transfer.py \
  --source /path/to/existing/region --output /path/to/separate/drive_validation
```

它执行真实 Drive 导出、下载、入库和清理，再请求同块像素，要求 valid 掩膜及
归一化 float16 数据逐值一致，结果写入 `validation.json`。此测试验证数据合同，
小块导出耗时不能用来推算整个区域的加速倍数。

独立验证掩膜选片与原完整波段选片的一致性：

```bash
python tools/data/validate_region_selection.py \
  --source /path/to/existing/region --output /path/to/separate/selection_validation \
  --limit 8 --workers 4
```

读取原 preparation.sqlite 的代表窗口，用独立掩膜缓存重做选片，逐项比较入选资产、
时间、有效比例和接受状态；另检查三个传感器各一个完整空间块的掩膜。
输出中的导出字节比较仅针对该窗口子集，不等于完整区域下载加速倍数。
# Landsat L2 与本地第三通道

新增样本 L2 输入包和 `--backend local` 区域准备，使用方式与输入兼容性见
[L2_INPUT_COLLECTION.md](L2_INPUT_COLLECTION.md)。旧 pixels/Drive 的 L1 数据合同保持不变。
