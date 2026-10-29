# 鄱阳湖 region2 采集与推理性能排查（2026-10-30）

## 边界与采集

用户指定资产：
[`users/xiaozhen6666666/ChinaWater/Poyang_Lake_region2`](https://code.earthengine.google.com/?asset=users/xiaozhen6666666/ChinaWater/Poyang_Lake_region2)。
通过已授权 Earth Engine API 获取一个要素的完整几何，保存到
`configs/inference/regions/poyang_lake_region2.geojson`，无需运行时再次查询该资产。

- WGS84 范围约为 115.783300–116.750000°E、28.366700–29.750000°N。
- 不额外缓冲；EPSG:32650 平面面积 14,423.364 km²。
- 文件 SHA-256：`814caeae4670181a23ac9f87fe25a89544c999c507db2c0a540b9223fd9da82a`。
- 维持原采集时段 2017-04 至 2024-12，共 356 个目标周期，五个 8 天输入槽。
- 新网格 15,432 × 9,576 个 10 m 像元，每日期 63,203 个模型窗口。
- 新计划身份：`5e20b376a804a1c9f2e896373c953bb801f20dcf24b96bbaafbe946287b7bee7`。

新数据根目录：

```text
/fossfs/xiaozhen/HydroState/inference/poyang_region2_5x8_201704_202412_20261030
```

IO2 后台沿用 `scripts/run_region_download_io.sh` 与 `computePixels`：4 并发、
2 CPU 亲和性、0 GPU；2026-10-30 00:21 HKT 启动 PID `1053887`。
冻结源码 `fbef218d0bb3cc662b43d064d505c79ac146a27a`，回执为新根目录的
`submission.json`，日志为 `logs/prepare_pixels_20261030.log`。
00:27 HKT 已入库 560 个缓存产品、完成 225 次网络请求；这是共享原生缓存产品数，
不是独立训练样本数或完整目标日期数。运行状态以后以实时日志为准。

启动前在新边界内完成 2020-07-11 单窗口实采：6 次网络请求、30 个原生产品，
采集约 14.04 秒。逐数组审计通过，正式读取器接受该窗口；S1/S2 有有效观测，
Landsat 在该窗口全部缺测，不能宣称小试覆盖了有效 Landsat 情形。
小试回执位于 `poyang_region2_pilot_20261030/pilot_audit.json`；全量日志已记录
Landsat 原生产品存在有效像素。小试时间不能直接外推全域多年工期。

本地区域回归测试 20 项通过，相关 Ruff、Shell 语法、CLI 参数和差异检查通过。
旧 HydroLakes 数据与正在运行的推理任务 `4360562` 均未覆盖或停止。

## 已发现的性能问题与证据边界

这些是代码审计发现的重复工作，尚不是各阶段耗时百分比：

1. `datasets/regional.py` 每窗口读取六组、五槽、三个数组，即不跨块时至少 90 次
   Zarr 数组切片。`storage/reader.py` 只缓存 group 对象，未缓存解压后的块；
   相邻重叠窗口重复访问相同压缩块，跨块时读取更多。共享文件系统缓存命中率尚未实测。
2. shared-LoRA 的注意力主干只遍历一次，但 `multilevel_encoder.py` 仍按七个任务分别
   归一化、恢复共享 token，`temporal_model.py` 再分别恢复观测序列。
   `mtlora.py` 还在七任务 × 三层上重复调用共享输入质量融合模块。
   这些可考虑复用；各任务具有独立参数的查询和解码器仍有必要独立计算。
3. 旧启动测试按 batch 大小截取探测记录前 N 个窗口，N 增大时样本组成会变化，
   甚至纳入另一个日期。因此先前 2/4/8/16 的吞吐不能作为严格公平的 batch 排名。
   新诊断要求所有 batch 大小遍历相同完整窗口池，并记录峰值显存与分段时间。

补充只读输入审计：在两期相同索引 27000/27001 上，均主要为五槽 S1、一槽 S2，
Landsat 全缺测；2020 年另有极少量 S2 有效像素。因此不能仅凭这四个窗口的
观测量解释两日期速度差，也不能据此代替 GPU profiling。
记录保存在旧根目录 `runs/profile_b58ca87_20261030/input_support_audit.json`。

## Debug GPU 诊断

- 作业：`4362058`；`c_foss_gpu`、debug QoS、1 L40、4 CPU、32 GiB、25 分钟。
- 冻结源码：`b58ca8760c2776f96480762de06cb3f38fed4681`。
- checkpoint 与统计量沿用当前 shared-LoRA epoch 20。
- 两日期各 32 个相同空间窗口，batch 2/4/8/16/20/24，OOM 有记录并停止更大候选。
- 区分原生读取、对齐、预处理/设备传输、预测、回传 CPU、拼接；附算子 trace。
- 比较 0/2/4 个读取 worker；不会写入正式区域产品。

截至 00:27 HKT 因 GPU 资源全部占用仍在排队。调度器暂估 10:49 HKT 开始，
不是承诺时间。报告将写至旧数据根目录：

```text
runs/profile_b58ca87_20261030/report/benchmark.json
runs/profile_b58ca87_20261030/report/operators.txt
runs/profile_b58ca87_20261030/report/model_trace.json
```

尚不能报告新的 GPU 实测结论、最大稳定 batch 或读取/计算各占多少。
原任务仍沿用其冻结代码和 batch 2；此次不修改运行中的推理策略。
