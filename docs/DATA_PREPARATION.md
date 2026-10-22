# 当前采样、准备、训练与验证

科学参数见 [流程规范](RETRIEVAL_PIPELINE.md)，代码职责见
[目录说明](REPOSITORY_STRUCTURE.md)。所有命令从仓库根目录、Python 3.12 环境执行。

## 1. 路径与采样

```bash
export HYDROSTATE_PYTHON=python
export HYDROSTATE_DATA_ROOT=/path/to/samples/retrieval_5x8_v1
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
已开始采样的目录绑定配置和代码，参数改变后使用新目录。

HPC 在联网 IO 节点执行 `scripts/run_gee_collection_io.sh`。
该脚本额外要求 `HYDROSTATE_CODE_ROOT`、`HYDROSTATE_SOURCE_HASHES`，并使用上述
Python、样本目录和候选目录；校验冻结代码后调用同一个 `run_gee_sampling.sh`。
不能将 WSL 的 `/mnt/d` 默认路径用于 HPC。

默认同时采集 4 个样本；IO 启动器仍限制为 2 CPU、单线程数学库，不使用 GPU。
每个传感器的五个时槽合成一次像素请求（最多 3 次输入请求/样本），
保留逐景 QA、逐日去重均值和各槽有效观测次数、时间统计。
工作线程完成即补位；在途样本也预留频率配额、位置日期及邻近图块约束，
样本不合格时释放预留，网络失败仍报错并保留断点。

SM 监督先用原有规则检查图块陆地比例，不合格时仅跳过 SMAP 请求；
合格图块依次检查 SMAP 网格及近似 33 km 支持范围，任一失败便跳过后续检查。
产品元数据、同格同周期 SM 观测和地形统计使用带锁缓存；覆盖质量缓存还绑定完整
GEE 覆盖表达式（来源影像、日期、阈值和投影），不会仅凭空间邻近复用。
所有覆盖和 ET 标签的计算规则保持一致。

## 2. 审计与训练统计

```bash
python tools/validate_zarr.py "$HYDROSTATE_DATA_ROOT" --pixel-limit 64
python tools/data/compute_target_statistics.py --config configs/hydrostate_mtlora.py   --output "$HYDROSTATE_DATA_ROOT/target_stats.json" --num-workers 4
```

审计默认检查全部分片属性及 64 个样本的数组；`--pixel-limit 0` 检查全部数组。
只有试采审计才使用 `--allow-partial`，正式准备要求七组完成标记。
统计仅使用训练划分和实际监督掩膜，已有统计文件不覆盖。

准备 MTLoRA / shared-LoRA 成对运行时可统一执行：

```bash
python tools/data/prepare_training.py prepare   --data-root "$HYDROSTATE_DATA_ROOT"   --statistics "$HYDROSTATE_DATA_ROOT/target_stats.json"   --run-root /path/to/run --num-workers 4
```

该入口检查完成数据、生成或验证统计、检查两个模型的读取和数据一致性，
保存展开配置及 `provenance/ready.json`。它不提交训练。

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
