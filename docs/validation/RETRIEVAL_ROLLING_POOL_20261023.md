# 滚动候选池、续采与扩容验收（2026-10-23）

生产提交：`0c547015275f86df77b0ee4bcaed6e1103f7944b`。采样、输入、标签和 QA 定义未改变；
像素生产函数及依赖签名与旧冻结版本一致：
`ecaaf86046bab4f70ab914277a7b407ae6bceb97f8623bbd398736e7fb1b9df1`。

## 实现

- 七个 JRC 池各最多 128 个候选、低水位 32；冰雪池最多 16、低水位 4。
  目标数量增大不放大池容量。最后接近配额时只补少量候选，已剩余候选保存供后续扩容。
- 查询与下载共用 4 个任务名额，不存在 JRC 内层四线程池。
  有候选的组直接下载，缺候选的组单独补充，不等待全部八组填满。
- `sampling_state.json` 保存队列、JRC 游标、冰雪搜索状态及在途操作。
  源候选缓存继续复用；失败请求重试，SIGTERM/SIGINT 停止新提交并等待在途写盘。
- `collection.json` 和原分片指纹不变。`collection_plan.json` 单独保存目标数量及执行参数。
  同目录提高 `--samples` 后按八组已有成功数补缺口，不改旧样本的空间划分。
- 完成版本的清单、标记和常用统计文件保存在 `releases/`，图像不复制；新数据版本
  重新计算训练统计量。QA、标签或影像生产定义变化不能作为调度更新直接续采。
- 此次从旧调度切换使用 `--adopt-from`，核对旧完整生产哈希与新旧像素生产签名；
  后续同定义续采不再需要重复迁移。

## 测试

102 项相关测试通过，26 条依赖弃用警告；静态检查、diff 检查和离线入口预览通过。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/retrieval tests/data/test_gee_collection.py \
  tests/test_mtlora.py tests/test_variable_packs.py -q
```

覆盖固定容量、低水位补充、四任务共享、慢查询不阻塞样本、全池被时空约束排除后
继续补点、候选源耗尽、游标恢复、失败查询重放、停止后续采、扩容补配额、
旧配置不变及完成版本快照、旧生产代码校验和不兼容数据变更拒绝。

## HPC 原目录切换

香港时间 17:54:08 向旧进程发送中断，17:54:28 确认旧进程退出并取得数据目录锁。
旧运行共保存 **327 个成功样本**。17:54:29 从新冻结代码启动，PID `330610`，
仍为 4 并发、2 CPU、0 GPU；目标保持 20,000，不自动提交训练。

数据目录继续为：
`/scr/u/xiaoz/HydroState/samples/retrieval_eight_20000_06aba36`。
目录后缀记录最初的建库版本；当前运行代码由运行记录和独立冻结工作区确定。

| 检查 | 结果 |
|---|---|
| 原始成功记录 | 327 个全部逐文件哈希一致 |
| 原始 collection.json | 逐文件哈希一致 |
| 第一个新增样本 | 重启后 27.96 秒 |
| 17:56:29 成功总量 | 347 个，新增 20 个 |
| 启动后约两分钟吞吐 | 599.6 个新增样本/小时，仅短时观测 |
| 当前发布清单审计 | 338 个分片契约、空间/日期限制通过 |
| 数组审计 | 64 个样本通过；所有发布的冰雪中心检查通过 |
| 活跃任务 | 4 |
| 实测池容量 | JRC 每组 ≤128，冰雪 ≤16 |
| 新运行日志警告/错误 | 0 |

这次复用了旧进程已生成的候选缓存，因此 28 秒产出首个新增样本不是冷启动性能承诺。
移除了“先准备 52,500 个候选”的屏障；总吞吐仍需在持续运行中评估。
审计时使用的是定期发布清单（338 个），随后成功记录增至 347 个，属于正常的发布间隔。

当前记录：`/scr/u/xiaoz/HydroState/submissions/collection_retrieval_eight_20000_20261023.json`。
新日志：`/scr/u/xiaoz/HydroState/submissions/collection_retrieval_eight_20000_rolling_20261023.log`。
切换证据：`/scr/u/xiaoz/HydroState/submissions/switch_retrieval_rolling_20261023.json`。
验收报告：`outputs/operations/20261023_rolling_validation.json`；HPC 同步保存为
`/scr/u/xiaoz/HydroState/submissions/retrieval_rolling_validation_20261023.json`。
