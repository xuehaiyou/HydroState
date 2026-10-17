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
  region_plan.py          根据缺失缓存生成有大小上限的区域矩形
  state.py                exports.sqlite 持久化导出状态
  backends/
    pixels.py             小请求 computePixels；兼容旧区域像素读取
    drive.py              区域导出、Range 续传、校验、入库和精确清理
  workflows/
    samples.py            样本规划与获取；保留旧 Drive 样本实现
    region.py             区域准备、缓存复用和日期清单
    drive_queue.py        有并发数和字节数上限的导出队列
hydrostate/storage/scene_cache.py  离线 Zarr 读写，无网络请求
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
在本项目 HPC 上直接运行于可联网的登录节点，不提交 Slurm 下载作业。

默认复用 `~/.config/earthengine/credentials`；授权需要包括 Drive 权限，
对应 quota project 必须启用 Drive API。也支持 `--drive-token /private/path/token.json`
以及 `--drive-quota-project`。密钥、token、数据和运行记录不进入 Git。

每个区域目录有唯一命名的 Drive 临时文件夹，程序拒绝同名文件夹歧义。
配置默认最多两个导出、两个文件下载线程，单导出原始数据最多 0.5 GiB，
待处理原始数据总额最多 1 GiB，另外保留 0.5 GiB Drive 空间。
这适配当前账号较小的剩余空间；扩容后可调整这些参数。
每个 TIFF 部分最多 3840×3840 像素，对齐 256 像素 GEE shard 和 480 像素缓存块。
这只是传输期间的 GeoTIFF 切块，最终 Zarr 不使用 ZIP。

当前版本下载时间上下文内与窗口相交的候选影像区域，再在本地按原质量规则选择。
没有在 GEE 端先筛选最优窗口，也没有改成时间合成影像。有效性单独导出为 valid 波段，
空像素显式为零；关闭 skipEmptyTiles，从而能严格区分空区域与尚未下载的 TIFF 部分。
这减少小请求往返，但仍可能导出最后未选中的候选影像；加速幅度需要实测。

## 恢复与数据安全

```text
区域根目录/
  plan.json / windows.parquet / catalogs/  原区域计划与目录
  input_cache/<sensor>/<scene>.zarr/       已归一化输入，跨日期复用
  manifests/<date>.parquet                 完整日期窗口选择结果
  preparation.sqlite                      可恢复的逐窗口结果
  exports.sqlite                          导出身份、task ID、文件、提交状态
  staging/drive/<export_id>/               临时 TIFF 和 .part 续传文件
  logs/ / submissions/                    启动日志与版本记录
```

- 区域像素和 Drive 工作流共用 `prepare.lock`，不会同时写入。
- 已有兼容缓存的 done 块直接复用；缓存网格、归一化和块尺寸必须一致。
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
