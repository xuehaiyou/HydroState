# 当前采样、准备、训练与验证

科学参数见 [流程规范](RETRIEVAL_PIPELINE.md)，代码职责见
[目录说明](REPOSITORY_STRUCTURE.md)。所有命令从仓库根目录、Python 3.12 环境执行。

## 1. 路径与采样

```bash
export HYDROSTATE_PYTHON=python
export HYDROSTATE_DATA_ROOT=/path/to/samples/retrieval_5x8_eight_groups
export HYDROSTATE_CANDIDATE_ROOT=/path/to/candidates/retrieval_5x8_v1
export HYDROSTATE_MODEL_PATH=/path/to/OlmoEarth-v1_2-Base
export GEE_PROJECT=cropland-477906

bash scripts/run_gee_sampling.sh --dry-run
bash scripts/run_gee_sampling.sh
```

统一参数文件为 `configs/data/collection/retrieval.py`。
`--dry-run` 只解析参数和生产代码哈希，不创建输出目录或连接 GEE。
可用 `--samples`、`--output`、`--candidate-dir`、`--project`、`--concurrency` 覆盖参数；
`COLLECTION_CONFIG` 可指定另一份完整的当前配置，不接受旧 `_base_` 采样配置。
数据定义、分片指纹固定；目标数量、并发和候选池容量独立保存为运行计划。
目标数量只允许增加；改变 QA、标签或影像定义仍需新目录。

HPC 在联网 IO 节点执行 `scripts/run_gee_collection_io.sh`。
该脚本额外要求 `HYDROSTATE_CODE_ROOT`、`HYDROSTATE_SOURCE_HASHES`，并使用上述
Python、样本目录和候选目录；校验冻结代码后调用同一个 `run_gee_sampling.sh`。
不能将 WSL 的 `/mnt/d` 默认路径用于 HPC。

默认候选查询和样本采集合计最多 4 个任务；IO 启动器仍限制为 2 CPU、单线程数学库，不使用 GPU。
JRC 每组 `candidate_pool_capacity=128`、`candidate_low_watermark=32`；
冰雪组容量 16、低水位 4。池容量不随目标总量增长；末期只按剩余配额补充。
候选来自打乱的全球块，小池内混合抽取；单块查询与像素下载在同一个线程池滚动补位。
每次提交前选择日期，将在途任务计入年份、月份、气候区和位置日期限制；
失败释放全部预占。默认不对已经下载的成功样本做 reservoir 替换。
可先用 `--samples 160 --concurrency 4` 试采，检查报告中的 `year_month_counts`、
`geographic_block_counts` 和 `valid_label_sample_counts` 后再扩大采集。
每个传感器按产品分辨率分组，每组五个时槽合成一次像素请求
（最多 6 次输入请求/样本；S1 一组、S2 三组、Landsat 两组）。
在 GEE 完成逐景 QA、逐日去重均值，按 10/15/20/30/60 m 保存波段与对应
有效日期数、时间统计；本地数据读取时才统一到 10 m。
完整五槽的光谱数组元素数比全部保存为 10 m 减少约 64%；这不是压缩文件大小
或总采样耗时的保证，请求数增加后的实际吞吐需实测。
工作线程完成即补位；在途样本也预留采样组配额、位置日期及邻近图块约束，
样本不合格时释放预留，网络失败仍报错并保留断点。

SM 监督先用原有规则检查图块陆地比例，不合格时仅跳过 SMAP 请求；
合格图块依次检查 SMAP 网格及近似 33 km 支持范围，任一失败便跳过后续检查。
产品元数据、同格同周期 SM 观测和地形统计使用带锁缓存；覆盖质量缓存还绑定完整
GEE 覆盖表达式（来源影像、日期、阈值和投影），不会仅凭空间邻近复用。
所有覆盖和 ET 标签的计算规则保持一致。
新格式使用 `retrieval_5x8_eight_groups` 样本目录，旧试采样本不会自动转换或混写；
JRC 中心点候选池仍可复用。冰雪候选缓存在独立 `dw_ice_v1` 子目录；
位置绑定实际 PML 日期，最终中心冰雪覆盖 ≥50% 才下载输入。每组成功 20 个构成 160 个试采。
`sampling_group_counts` 是八组硬配额，`frequency_counts` 是实际 JRC 频率组成。

## 原目录续采与扩容

相同目标重新执行采样命令即可恢复。增加总量时保持同一数据目录及数据定义：

```bash
# 例如从 20,000 扩到 40,000；八组各从 2,500 补到 5,000。
bash scripts/run_gee_sampling.sh --samples 40000 --concurrency 4
```

HPC 同样通过冻结代码的 `run_gee_collection_io.sh` 执行，并先确认旧采样进程退出。
`collection.json` 保留原始生产配置，`collection_plan.json` 的数量才是当前目标。
候选余量、源游标和在途请求保存在 `sampling_state.json`，缓存可继续复用。
SIGTERM/SIGINT 会等待在途写盘后退出；失败请求在下次运行时重试。
已经完成的版本在扩容前保存到 `releases/`，旧统计量保留；为扩容结果指定新的
统计文件，例如 `target_stats_n40000.json`。使用旧清单训练时，`manifest` 指向快照，
`data_root` 仍指向原样本根目录，不能将相对分片路径误解为快照目录中的文件。

从本次修改前的冻结版本切换时，仅第一次增加：
`--adopt-from /path/to/previous/frozen/code`。工具会核对旧完整生产哈希，并确认
输入/标签函数与其依赖的生产签名一致；不会改写旧分片或放行 QA 变化。

## 2. 审计与训练统计

```bash
python tools/validate_zarr.py "$HYDROSTATE_DATA_ROOT" --pixel-limit 64
python tools/data/compute_target_statistics.py --config configs/hydrostate_mtlora.py   --output "$HYDROSTATE_DATA_ROOT/target_stats.json" --num-workers 4
```

审计默认检查全部分片属性及 64 个样本的数组；`--pixel-limit 0` 检查全部数组。
只有试采审计才使用 `--allow-partial`，正式准备要求八组完成标记。
统计仅使用训练划分和实际监督掩膜，已有统计文件不覆盖。

准备 MTLoRA / shared-LoRA 成对运行时可统一执行：

```bash
python tools/data/prepare_training.py prepare   --data-root "$HYDROSTATE_DATA_ROOT"   --statistics "$HYDROSTATE_DATA_ROOT/target_stats.json"   --run-root /path/to/run --num-workers 4
```

该入口检查完成数据、生成或验证统计、检查两个模型的读取和数据一致性，
保存展开配置及 `provenance/ready.json`。它不提交训练。
使用独立标签包时传 `--config configs/hydrostate_extended.py`；新标签声明、
发布命令和环境变量见 [扩展指南](EXTENDING_VARIABLES.md)。已有统计不能用于不同标签包版本。

## 3. 模型选择和验证

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_mtlora.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_mtlora_shared.py
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/hydrostate_mtlora.py CHECKPOINT.pth
```

两个训练命令分别运行，并使用不同 `HYDROSTATE_WORK_DIR`。
每个任务只用一张 GPU；当前默认 microbatch=2、累积=8，尚未宣称吞吐最优。
`tools/predict.py` 使用测试循环，不是当前七输出的区域地图导出器。

## 4. HPC 资源包装

当前 Slurm 文件只有：

- `scripts/slurm/prepare_retrieval.sh`：6 CPU，执行同一个 Python 准备入口。
- `scripts/slurm/train_retrieval.sh mtlora|lora`：1 GPU、6 CPU，验证数据后调用 `scripts/train.sh`。

冻结发布设置：`HYDROSTATE_CODE_ROOT`、`HYDROSTATE_CODE_COMMIT`、
`HYDROSTATE_SOURCE_HASHES`；数据与实验设置：`HYDROSTATE_DATA_ROOT`、
`HYDROSTATE_TARGET_STATS`、`HYDROSTATE_MODEL_PATH`、`HYDROSTATE_RUN_ROOT`、
`HYDROSTATE_WORK_DIR`。Conda 路径可通过 `HYDROSTATE_CONDA_BASE` 和
`HYDROSTATE_CONDA_ENV` 指定。分区与数据路径按所在集群选择。

`tools/watch_mtlora_collection.py` 是显式启动的调度器，等待完成标记后提交准备和成对训练，
保留提交意图以防重复。采样入口不会自动启动它。
旧配置/脚本已移到 `configs/experiments/legacy`、`scripts/legacy` 和 `tools/legacy`。
