# 仓库结构与职责

正式路径：**采样配置 → retrieval 采样 → 严格读取/审计 → 训练统计 → 模型配置 → 训练/验证**。
运行数据、缓存和 checkpoint 放在代码仓库之外。

```text
configs/
  data/collection/retrieval.py   当前采样参数
  hydrostate_retrieval.py        公共训练配置，从数据契约导入任务和输入定义
  hydrostate_mtlora*.py          模型选择及消融
  _base_/runtime.py              MMEngine 通用运行设置
  experiments/legacy/           历史架构、数据和机器配置
hydrostate/
  data/retrieval/                当前产品监督流程
  data/gee/                     GEE 产品、传输、请求和历史区域工作流
  data/spatial.py                公共网格与地理划分
  datasets/                     当前读取器、拼批和采样器
  storage/                      与任务无关的数组描述、Zarr 读写和缓存
  tasks.py                      任务支持范围校验；默认解析当前七任务
  statistics.py                 通用流式统计
  models/, losses/, evaluation/ 模型、损失及指标；架构仍由 config 选择
  legacy/                       历史采样器、读取器、schema 和统计
  data/regional/                正式区域合成规划、直接像素/Drive 准备；原单景实现保留原位
  data/local/                   本地历史影像准备组件
  inference/                    正式七任务区域导出及已有三变量导出
  experiments/                  专项实验组件
scripts/
  run_region_download_io.sh      当前区域输入准备包装
  run_gee_sampling.sh            唯一本地采样包装入口
  run_gee_collection_io.sh       HPC IO 环境检查后调用同一采样入口
  train.sh                      单 GPU 训练运行器
  slurm/                        当前准备/训练/区域推理资源声明
  legacy/                       历史调度和采样命令
tools/                         薄 CLI；data/ 为当前准备，legacy/ 为历史命令
docs/                          当前文档；history/ 为历史记录
tests/                         共享模块及历史回归；retrieval/ 为当前流程检查
```

## 当前采样包

- `contract.py`：唯一当前版本、任务、输入配置和数组形状。
- `specs.py`：产品来源、波段/分辨率/单位、归一化和标签监督规格；契约从这里派生。
- `configuration.py`：训练配置与该契约的一致性检查。
- `cli.py`：解析采样参数，离线预览，绑定生产代码哈希。
- `catalog.py`：候选点和影像元数据缓存；`sampling.py`：配额与时空去重。
- `candidates.py`：可恢复的 JRC 块游标；`scheduling.py`：固定容量池、低水位补充，
  逐任务选日期并进行软均衡；查询与下载共用线程池，保存队列和在途操作。
- `plan.py`：不变生产契约、可扩展目标、生产签名核验及清单快照。
- `ice_candidates.py`：独立 DW 冰雪候选，绑定目标日期、缓存空查询并共享四任务并发。
- `composites.py`：逐景 QA、均值合成、观测质量摘要。
- `alignment.py`：产品分辨率网格定义、本地对齐 10 m 和跨组有效性合并。
- `gee_labels.py`：产品读取；`labels.py`：监督掩膜与面积聚合。
- `collector.py`：输入提取、样本提交、续采和完成标记。
- `label_packs.py`：独立标签包发布、按样本 ID 关联、时间/单位/网格校验及审计。
  复用公共 `storage/packs.py`，当前命令是 `tools/data/build_label_pack.py`。
- `validation.py`：数组检查；`audit.py`：已发布数据审计。
- `statistics.py`：训练集监督尺度上的统计；`release.py`：成对训练准备。

公共工具不从历史采样器、历史读取器导入私有对象。
`storage/reader.py` 的 `ShardCache`、`datasets/collate.py` 的 `hydro_collate`、
`data/spatial.py` 的网格函数供不同流程复用。分片 writer 必须显式接收 schema。

## 历史边界

旧四槽/三变量 schema 使用 `LegacySchema`，旧读取器使用
`hydrostate.legacy.dataset.HydroStateZarrDataset`；由历史配置显式注册。
共享模型中的历史默认参数明确命名为 `LEGACY_TASKS`，正式配置始终绑定当前七任务。
历史工具使用 `hydrostate.legacy.statistics`，当前统计入口只接受当前契约。

历史模型实现继续保留，避免把“清理旧数据流程”误变成“删除模型对比能力”。
归档配置继续加载其历史数据定义；不能直接用它们对当前七任务样本训练。
当前区域入口为 `tools/prepare_region.py` 和 `tools/predict_region.py`，不在 legacy 下。
`regional/plan.py` 定义计划，`exports.py` 规划共享合成请求；`pixels.py` 调度直接像素获取，
`reuse.py` 核对跨区域原生图块并复用完整覆盖，避免重复网络下载；
`pipeline.py` 调度与恢复 Drive 导出；
`storage/regional.py` 校验入库，`datasets/regional.py` 读取五槽，`inference/regional.py`
加载当前 checkpoint 并拼接七任务及总 ET。原有四槽/单景文件保留原位供已有实验追溯，
正式流程不调用其选片或缓存实现，也不依赖 LegacySchema。
`inference/regional_runtime.py` 负责输入抽查、GPU batch 实测和运行回执；
`regional_benchmark.py`、`regional_comparison.py` 分别测量流水线和固定旧版本输出对照；
`storage/spatial_cache.py` 提供有字节上限的原生数组解压块缓存；
`regional_reports.py` 负责物理范围检查和仅用于显示的 PNG 掩膜。
`scripts/slurm/predict_region.sh` 是单卡 FossFS 推理入口，不承担数据或拼接逻辑。

新增业务逻辑放入 `hydrostate/`，`tools/` 只负责参数和调用；
Shell 负责资源、环境和进程启动，数据检查统一调用 Python 服务。
