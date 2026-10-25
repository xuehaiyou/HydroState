# 2026-10-25 当前区域流程接入直接像素获取

## 已实现

`tools/prepare_region.py prepare` 默认使用当前五槽 `computePixels` 流程，
`--backend drive` 保留显式 Drive 通道。原始分辨率、QA、时间支持、轨道选择、
归一化、产品身份和模型读取不变，两种传输调用同一合成表达式和原子入库。

全域默认 4 个获取/入库线程，另有 1 个有界规划线程；请求不超过 32 MiB，
最多 3 次暂时性错误尝试，scratch 预留 20 GiB。确认无源影像的请求在本地写入
全缺测产品；计算失败和无效数组不能成为空观测。每个完成请求记录耗时及有效像素量。

`pixels.json` 与 `drive.json` 为运行设置，不进入数据生产指纹。旧的当前五槽目录可
重新执行相同科学配置的 `plan` 添加直接通道设置，不改写计划与成功产品。
跨产品版本的历史四槽数据不在此复用范围。

切换先停止旧本地调度，再执行 `--backend drive --drain`，接收并清理已记录的
在途导出。直接通道持有同一个准备锁，并拒绝在 Drive 记录尚未清空时启动。
恢复同时处理已保存实际任务 ID、只保存请求 UUID 两种情况，优先查找原任务，避免重复提交。

## 本地验证

- `pytest tests/regional tests/retrieval/test_gee_expressions.py -q`：19 项通过。
- 增加请求 UUID 恢复分支后，四个 Drive 恢复/交接组合全部通过。
- 另覆盖请求预算、独立运行配置、计划身份不变、空源与请求失败的区分、
  部分成功后续跑不重写缓存、暂时性错误与永久错误的重试差异。
- 新建和重新加载计划统一为 JSON 类型，修正初次运行返回 tuple、恢复为 list 时
  可能误报冻结目标变化的问题；序列化指纹和已存在计划不变。
- 相关 Ruff、Shell 语法、差异空白检查通过。

## 真实入口试采

使用 `configs/inference/poyang_pilot.json`，目标日期 2020-07-11，1 个 1.2 km 模型窗口。

- 新目录：`/mnt/d/Hydrostate/inference/poyang_pixels_switch_pilot_20261025`。
- 参照目录：`/mnt/d/Hydrostate/inference/poyang_5x8_pilot_20261025`（此前 Drive 试采）。
- 完整准备入口耗时 37.51 s，完成六组五槽共 30 个产品并发布日期。
- 两目录的计划身份相等；`RegionalDataset.native_inputs()` 返回的 22 个输入数组
  完全相等，窗口通过可推理性筛选。
- 本区域 Landsat 的共同掩膜为空，验证其缺测一致性，不宣称具有有效 Landsat 观测。
- 试采记录：`outputs/operations/20261025_pixels_switch_pilot.json`。

这次是正式入口的小区域实采，与此前完整 9.6 km 块的
[传输对照](REGIONAL_PIXELS_BENCHMARK_20261025.md)互补，不能据此给出全湖多年任务的固定 ETA。

## HPC 交接记录

部署的 commit、旧进程停止、Drive 交接、新进程 PID 和启动检查保存在具体运行回执中，
以对应输出根目录的 `submissions/`、日志和 `pixels_progress.json` 为准。
生产数据目录沿用：

`/scr/u/xiaoz/HydroState/inference/poyang_retrieval_5x8_201704_202412_f9fce12`

目录名中的旧提交号是初次创建来源；后续运行代码版本由每次提交回执确定。
