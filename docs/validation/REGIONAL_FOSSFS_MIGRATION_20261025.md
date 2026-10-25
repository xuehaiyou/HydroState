# 2026-10-25 区域输入迁移至 FossFS

2026-10-25 22:06 HKT 完成迁移，并从新目录恢复 `computePixels` 采集。
旧 SCR 目录及删除暂存目录均已删除，不保留旧路径的符号链接。

## 位置与运行记录

- 原数据根目录：`/scr/u/xiaoz/HydroState/inference`。
- 新数据根目录：`/fossfs/xiaozhen/HydroState/inference`，包含原目录下的当前与历史区域数据。
- 当前鄱阳湖根目录：
  `/fossfs/xiaozhen/HydroState/inference/poyang_retrieval_5x8_201704_202412_f9fce12`。
- 实际执行代码：`535d29eb8822e63144bfa9fada754f28c2ab9251`，冻结源码位于
  `/scr/u/xiaoz/HydroState/submissions/region_pixels_535d29e_20261025`。
- IO2 采集 PID：`632873`，4 个获取线程、2 CPU、0 GPU。运行状态需查实时进程和日志。
- 完整迁移回执：`/fossfs/xiaozhen/HydroState/logs/inference_migration_20261025.json`，
  最终 `phase=complete`；当前运行入口记录在新根目录的 `submission.json`。

采集和模型推理均直接传入新 `--root`。目录名的 `f9fce12` 是最初建库来源，
实际执行版本以运行回执为准。此次操作只修改数据位置、运行路径和历史缓存引用。

## 完整性与续跑验证

1. 在线预复制后停止原采集进程、持有原准备锁，补齐最后增量。
   最终同步与完整 checksum 比较覆盖 20 个互不重叠的目录分区，最大同时运行 4 路。
2. 文件内容和链接目标比较一致。比较另发现 806 处目录修改时间差异，
   已逐项按源目录校正并复核；这些差异不涉及数组或 JSON 文件内容。
3. 原 `plan.json` 的 SHA-256 保持一致，计划身份仍为
   `13c48f6819117b88791b68128ca1881acead6bfc983eb24b926cddc0186e1188`。
   原有 **10,120 个成功产品标记**逐项校验一致，保留 **11 个完整目标日期**。
4. 使用正式 `RegionalDataset.native_inputs()` 读取首个已完成日期的第
   `0 / 16910 / 33820` 个窗口。三个窗口均有效，每个窗口的 **22 个原生输入数组**
   在迁移前后形状、dtype 和逐字节 SHA-256 一致。
5. 历史 `poyang_centered8d_201704_202512` 的 `catalogs`、`input_cache`、
   `quality_cache` 三个共享缓存链接及 `cache_reuse.json` 的位置字段已改到 FossFS，
   并验证链接目标存在。历史提交回执中的旧路径保留为迁移前的运行证据。
6. 在新根目录重新启动原冻结版本，确认新的真实网络请求、有效像素和成功标记后才删除旧副本。
   删除完成时的进度快照记录新增 157 个产品、105 次网络请求，采集状态为 `running`。
   随后再次确认旧根目录、复制暂存目录、删除暂存目录均不存在，且进程参数直接使用新目录。

这里验证的是存储迁移、正式输入读取和采集续跑；完整区域采集仍在进行，模型推理未由此次迁移启动。
启动命令与当前存储路径见 [区域推理指南](../REGIONAL_INFERENCE.md)。
