将 GEE 样本上传到 HPC2021 并微调 HydroState
===========================================

采用“固定清单 → 上传其引用的 Zarr → 校验 → 单卡训练检查 → 四卡训练检查 → 正式微调”的顺序。
采样可以继续运行；每次实验读取同一份清单，保留原来的 train/validation/test 划分。

本文沿用仓库原有配置：登录账号 `xiaoz@hpc2021.hku.hk`，代码位于
`/home/xiaoz/HydroState`，大文件位于 `/fossfs/xiaozhen/HydroState`，
Slurm account 为 `geog_geors`，partition 为 `c_foss_gpu`，qos 为 `gpu`。
2026-09-22 已通过只读 SSH 登录确认上述路径和 GPU 分区，节点为 NVIDIA L40；
Conda 在 `/home/xiaoz/miniconda`，登录节点 glibc 为 2.28。
WSL 和 Windows 的用户公钥指纹相同，但只有 Windows 的 `known_hosts` 有 HPC 记录。
下面的本地命令直接复用该可信记录，同时保留严格主机密钥校验；不需要复制私钥。
远端已有 `hydrostate` 环境（Python 3.12.14、torch 2.7.1+cu128、torchvision
0.22.1+cu128）和约 983 MB 的 OlmoEarth 权重；当前项目要求 torch 2.9，需按第 3 节
更新训练依赖。权重同步使用 checksum 比较，已有文件内容一致时不会重新传输。

**1. 本地 WSL：固定本次上传清单**

以下命令在 `/home/xiaoz/HydroState` 执行。`gee_20260922` 是本次快照名称；
以后扩大样本集时使用新名称。第一次只运行一次本节，断线续传时直接重跑第 2 节。

```bash
cd /home/xiaoz/HydroState
export HS_SNAPSHOT=gee_20260922
export HS_STAGE="$PWD/transfer/$HS_SNAPSHOT"
export HS_SOURCE=/mnt/d/Hydrostate/samples
export HS_REMOTE_DATA="/fossfs/xiaozhen/HydroState/datasets/$HS_SNAPSHOT"
export RSYNC_RSH='ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/mnt/c/Users/zhenxiao/.ssh/known_hosts -o IdentitiesOnly=yes -i /home/xiaoz/.ssh/id_rsa'

/home/xiaoz/miniconda3/envs/hydrostate/bin/python - <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import pandas as pd

source = Path(os.environ['HS_SOURCE'])
stage = Path(os.environ['HS_STAGE'])
table = pd.read_parquet(source / 'samples.parquet')
assert len(table) and not table.sample_id.duplicated().any()
assert {'train', 'validation', 'test'} <= set(table.split)
paths = sorted(set(table.input_shard) | set(table.label_shard))
for value in paths:
    path = PurePosixPath(value)
    assert not path.is_absolute() and '..' not in path.parts
    assert path.parts[0] == 'zarr', value
    assert (source / value / '_SUCCESS.json').is_file(), value

# Prevent accidentally replacing a snapshot already used by an experiment.
stage.mkdir(parents=True, exist_ok=False)
table.to_parquet(stage / 'samples.parquet', index=False)
(stage / 'zarr-files.txt').write_text(''.join(p + '/\n' for p in paths))
summary = {
    'samples': len(table),
    'splits': table.split.value_counts().to_dict(),
    'frequency_bins': table.frequency_bin.value_counts().sort_index().to_dict(),
}
(stage / 'snapshot.json').write_text(json.dumps(summary, indent=2))

# Preserve tested package versions and the installed Git commit IDs.
# Select torch/CUDA separately for the remote GPU driver.
freeze = subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze', '--exclude-editable'], text=True
)
(stage / 'environment.freeze.txt').write_text(freeze)
requirements = []
for line in freeze.splitlines():
    if not line or line.startswith('#'):
        continue
    name = re.split(r'==| @ ', line, maxsplit=1)[0]
    normalized = name.lower().replace('_', '-')
    if normalized in {'hydrostate', 'torch', 'torchvision', 'torchaudio', 'triton'}:
        continue
    if normalized.startswith('nvidia-'):
        continue
    # Conda sometimes records build-machine paths, e.g. packaging @ file://...
    if ' @ file:' in line:
        line = f'{name}=={md.version(name)}'
    requirements.append(line)
(stage / 'requirements-hpc.txt').write_text('\n'.join(requirements) + '\n')
print(json.dumps(summary, indent=2))
print('Snapshot:', stage)
PY
```

清单中的 `input_shard` 和 `label_shard` 是相对路径；保持 `zarr/...` 结构即可，
无需重写成 HPC 绝对路径。使用 `samples.parquet`，不要使用尚未全部落盘的
`sample_index.parquet`。这里读取的是原子发布的清单，引用的 shard 已经完成且不可变。

**2. 本地 WSL：上传代码、样本和预训练权重**

先登录检查并退出：

```bash
ssh -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/mnt/c/Users/zhenxiao/.ssh/known_hosts \
  -o IdentitiesOnly=yes -i /home/xiaoz/.ssh/id_rsa xiaoz@hpc2021.hku.hk
ls -ld /fossfs/xiaozhen
df -h /fossfs/xiaozhen
sinfo -p c_foss_gpu -o '%P %a %l %D %G'
exit
```

回到本地，沿用第 1 节的环境变量。只上传训练需要的目录和两个权重文件；
不上传 `downloads/`、`candidates/`、SQLite、Drive OAuth 文件或其他本地凭据。
代码同步会更新远端同名源码，先保存远端未提交的修改。

```bash
ssh -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/mnt/c/Users/zhenxiao/.ssh/known_hosts \
  -o IdentitiesOnly=yes -i /home/xiaoz/.ssh/id_rsa \
  xiaoz@hpc2021.hku.hk "mkdir -p \
  /home/xiaoz/HydroState \
  '$HS_REMOTE_DATA' \
  /fossfs/xiaozhen/HydroState/logs \
  /fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base"

rsync -avh --partial --exclude='__pycache__/' \
  hydrostate configs scripts tools docs README.md pyproject.toml \
  xiaoz@hpc2021.hku.hk:/home/xiaoz/HydroState/

# Stop this transfer block on failure, before publishing the manifest.
(
set -e
# With --files-from, recursive copying must be requested explicitly (-r).
rsync -arvh --partial --info=progress2 \
  --files-from="$HS_STAGE/zarr-files.txt" \
  "$HS_SOURCE/" "xiaoz@hpc2021.hku.hk:$HS_REMOTE_DATA/"

# Publish the manifest only after the Zarr transfer succeeds.
rsync -avh \
  "$HS_STAGE/samples.parquet" "$HS_STAGE/snapshot.json" \
  "$HS_STAGE/requirements-hpc.txt" "$HS_STAGE/environment.freeze.txt" \
  "xiaoz@hpc2021.hku.hk:$HS_REMOTE_DATA/"
)

rsync -avhc --partial --info=progress2 \
  pretrained/OlmoEarth-v1_2-Base/config.json \
  pretrained/OlmoEarth-v1_2-Base/weights.pth \
  xiaoz@hpc2021.hku.hk:/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base/
```

中断后重跑相应的 rsync 命令，保持同一份本地快照。不要在训练期间覆盖其远端清单。
大量 Zarr 小文件同步可能较慢；先完成这一版可恢复的传输，再根据实际 I/O 表现考虑打包。

**3. HPC：准备 Python 环境**

```bash
ssh -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/mnt/c/Users/zhenxiao/.ssh/known_hosts \
  -o IdentitiesOnly=yes -i /home/xiaoz/.ssh/id_rsa xiaoz@hpc2021.hku.hk
export HYDROSTATE_DATA_ROOT=/fossfs/xiaozhen/HydroState/datasets/gee_20260922
export HYDROSTATE_MODEL_PATH=/fossfs/xiaozhen/HydroState/pretrained/OlmoEarth-v1_2-Base
cd /home/xiaoz/HydroState

# This environment already exists on the inspected HPC account.
source /home/xiaoz/miniconda/etc/profile.d/conda.sh
conda activate hydrostate

# Check the actual GPU-node driver before choosing a CUDA wheel.
srun --account=geog_geors --partition=c_foss_gpu --qos=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --time=00:10:00 \
  bash -c 'getconf GNU_LIBC_VERSION; nvidia-smi'

# cu128 matches the local environment; use cu126 if required by the remote driver.
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$HYDROSTATE_DATA_ROOT/requirements-hpc.txt"
python -m pip install --no-deps -e .
python -m pip check
```

如果在另一台机器上没有 `hydrostate` 环境，先执行
`conda create -n hydrostate python=3.12 -y`。当前 HPC 已有环境，可直接激活并更新依赖。
requirements 保留了本地 OlmoEarth/rslearn 的具体 Git commit；`--no-deps -e .`
避免项目安装阶段再次拉取未固定的 Git HEAD。依赖安装需要 HPC 可访问包源和 GitHub。
训练使用本地权重及 Zarr，不需要 GEE/Drive 登录。

PyTorch 2.9.1 / torchvision 0.24.1 的 CUDA 12.6 和 12.8 wheel 均见
[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。
不能仅凭登录节点上 `nvcc` 的版本判断兼容性；在分配到的 GPU 节点上执行检查。
若出现系统 GLIBC 或 NVIDIA driver 不兼容，需使用集群支持的运行环境或容器，
不要直接把项目依赖降到不符合 `pyproject.toml` 的 torch 版本。

**4. HPC：核对上传数据**

```bash
python tools/validate_zarr.py "$HYDROSTATE_DATA_ROOT/samples.parquet" \
  --data-root "$HYDROSTATE_DATA_ROOT"

python - <<'PY'
import os
import pandas as pd
from mmengine.config import Config
root = os.environ['HYDROSTATE_DATA_ROOT']
table = pd.read_parquet(root + '/samples.parquet')
print('Samples:', len(table))
print(table.split.value_counts())
cfg = Config.fromfile('configs/hydrostate_hpc.py')
print('Training data:', cfg.train_dataloader.dataset)
print('Weights:', cfg.model.encoder.model_path)
PY
```

`validate_zarr.py` 校验清单引用、完成标记及数组形状；随后用 GPU 训练检查覆盖
模型加载、实际数据读取、损失、反向传播和优化器更新。

**5. HPC：先运行单卡、再运行四卡的短训练检查**

Slurm 会在脚本执行前打开日志文件，所以必须提前创建日志目录。
新脚本 `scripts/slurm/11_train_hpc.sh` 根据环境变量选数据和配置；继承项目现有
account/partition/qos，单节点四 GPU。可在 `sbatch` 命令行覆盖这些资源参数。

```bash
mkdir -p /fossfs/xiaozhen/HydroState/logs
export HYDROSTATE_CONFIG=configs/hydrostate_hpc_smoke.py
export HYDROSTATE_WORK_DIR=/fossfs/xiaozhen/HydroState/work_dirs/gee_20260922_smoke_1gpu
export HYDROSTATE_GPUS_PER_NODE=1

sbatch --export=ALL --job-name=hs_smoke1 --gres=gpu:1 \
  --cpus-per-task=8 --time=00:30:00 scripts/slurm/11_train_hpc.sh
squeue -u xiaoz
```

查看上述命令返回的作业号，用其替换 `JOB_ID`：

```bash
tail -f /fossfs/xiaozhen/HydroState/logs/hs_smoke1-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

单卡检查使用 batch size 1，运行 8 个 optimizer step，不运行完整验证集也不保存权重。
确认作业完成、损失有限且无 OOM 后，再检查实际四卡 DDP：

```bash
export HYDROSTATE_WORK_DIR=/fossfs/xiaozhen/HydroState/work_dirs/gee_20260922_smoke_4gpu
export HYDROSTATE_GPUS_PER_NODE=4
sbatch --export=ALL --job-name=hs_smoke4 --time=00:30:00 \
  scripts/slurm/11_train_hpc.sh train_dataloader.batch_size=2
```

四卡检查采用正式训练相同的每卡 batch size 2，确认通讯、显存及梯度同步可用。
模型的部分模态参数可能在一个批次中没有参与计算，因此 HPC 配置设置
`find_unused_parameters=True`；参数含义见
[MMEngine 官方文档](https://mmengine.readthedocs.io/en/v0.10.5/api/generated/mmengine.model.MMDistributedDataParallel.html)。

脚本默认 BF16。若分配的 GPU 不支持原生 BF16，设置
`export HYDROSTATE_PRECISION=float16`，然后重新运行单卡及四卡检查。
OOM 时先减小每卡 batch size；梯度累积不能减小单个 microbatch 的显存需求。
检查成功只证明执行路径可用，三项监督的有效数量和精度仍应分别监测。

**6. HPC：提交正式微调**

必须等待四卡短训练检查成功后执行。新建正式实验目录，避免继承 smoke 的输出设置。

```bash
export HYDROSTATE_CONFIG=configs/hydrostate_hpc.py
export HYDROSTATE_WORK_DIR=/fossfs/xiaozhen/HydroState/work_dirs/gee_20260922_v1
export HYDROSTATE_GPUS_PER_NODE=4

sbatch --export=ALL --job-name=hs_finetune scripts/slurm/11_train_hpc.sh
squeue -u xiaoz
```

初始设置：每卡 batch size 2，四卡，梯度累积 4 次，有效全局 batch size 32；
AdamW 基础学习率 1e-4，encoder 使用逐层学习率衰减，训练 50 epoch，
每 epoch 验证并按 `hydro_score` 保存最佳权重。这些继承项目的初始超参数，
并非已验证的最优值。此配置更新 encoder、共享 decoder 和三个任务 head。

输出目录包括训练日志、epoch checkpoint、最佳 checkpoint 和 `last_checkpoint`。
分开查看 water / soil_moisture / et 的物理单位指标，尤其注意 SMAP 有效标签较少。
当前部分数据可能还缺某些水体频率分层；它适合阶段性实验，不能据此宣称完整全球精度。

**7. HPC：恢复中断的同一实验、独立测试**

同一份数据、配置和 work_dir 的作业中断后，可恢复训练：

```bash
sbatch --export=ALL --job-name=hs_resume \
  scripts/slurm/11_train_hpc.sh resume=True
```

`resume=True` 会寻找该 work_dir 的 `last_checkpoint` 并恢复优化器/进度。
如果要明确指定 checkpoint，可在其后追加 `load_from=/absolute/path/epoch_N.pth`。
不要将扩大后的数据清单覆盖到正在恢复的实验；新增数据使用新快照和新 work_dir，
需要继续学习时加载既有模型权重，重新设计本次训练日程。

正式训练完成后，从 work_dir 中选择实际的 `best_hydro_score_*.pth` 路径。
测试仍需 GPU 分配，例如（替换 `BEST_CHECKPOINT`）：

```bash
srun --account=geog_geors --partition=c_foss_gpu --qos=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=02:00:00 \
  python tools/test.py configs/hydrostate_hpc.py BEST_CHECKPOINT \
  --cfg-options test_dataloader.batch_size=1 \
  work_dir=/fossfs/xiaozhen/HydroState/work_dirs/gee_20260922_test
```

测试脚本使用原有 test 划分；用 validation 选模型、用 test 做最终报告。
