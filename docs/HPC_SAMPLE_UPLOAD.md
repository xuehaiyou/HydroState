首批 1,024 个样本上传及后续续传
==============================

代码和权重已经在 HPC，本流程只上传样本。以下上传命令由用户在本地 WSL 手动执行。

本地首批清单已经生成在 `outputs/hpc_transfer/gee_20260922_n1024/`，包含
`samples.parquet`、`zarr-files.txt` 和 `snapshot.json`。按原有 split 与水体频率分层
抽取，保留原来的地理划分；采样程序继续收集数据不会改变这份清单。

**首次上传**

```bash
cd /home/xiaoz/HydroState
bash scripts/sync_hpc_samples.sh outputs/hpc_transfer/gee_20260922_n1024
```

脚本使用 WSL 用户私钥和 Windows 已保存的 HPC 主机记录。只传清单引用的 Zarr，
不传采样候选池、原始下载 TIFF、代码或训练权重。所有版本复用同一份远端 Zarr：

```text
/fossfs/xiaozhen/HydroState/samples_v1/
├── zarr/<sample_id>/inputs.zarr/
├── zarr/<sample_id>/labels.zarr/
└── manifests/
    └── gee_20260922_n1024/
        ├── samples.parquet
        └── snapshot.json
```

rsync 完成后，脚本调用远端仓库已有的 `tools/validate_zarr.py` 校验引用、完成标记
及数组形状，通过后才发布该版本的 `samples.parquet`。看到 `Published: ...` 表示完成。
远端已存在相同清单时允许重跑；存在同名但内容不同的清单时拒绝覆盖。

**网络中断后继续**

直接重跑上面的同一条命令。已经传完且大小、修改时间未变化的文件会跳过，
部分文件通过 rsync 的临时文件继续传输。不要重新生成或修改同名快照，
也不要删除远端 `zarr/` 或 `.rsync-partial/`。

**以后增加到 2,048 个样本**

先生成新版本；`--extend` 保证首批 1,024 个样本全部保留，原有 split 不变：

```bash
cd /home/xiaoz/HydroState
/home/xiaoz/miniconda3/envs/hydrostate/bin/python \
  tools/data/prepare_training_snapshot.py \
  --source /mnt/d/Hydrostate/samples \
  --output outputs/hpc_transfer/gee_n2048_v2 \
  --limit 2048 \
  --extend outputs/hpc_transfer/gee_20260922_n1024/samples.parquet

bash scripts/sync_hpc_samples.sh outputs/hpc_transfer/gee_n2048_v2
```

这里的 `--limit` 是新版本总样本数，不是额外增加的数量。远端共用 `zarr/`，
已上传的旧样本会跳过，只需传新增或未完成的文件。断线时重跑第二条上传命令即可，
不要重复执行生成已存在快照的命令。

要上传届时所有已完成样本，换一个新输出目录，并将 `--limit` 改成 `0`。
继续使用 `--extend` 指向上一版清单；采样计划中尚未完成的样本不会进入上传列表。

**训练引用方式及后续新增数据**

首批数据上传成功后，训练应明确指定共享数据根目录和该版本清单：

```text
data_root = /fossfs/xiaozhen/HydroState/samples_v1
manifest = /fossfs/xiaozhen/HydroState/samples_v1/manifests/gee_20260922_n1024/samples.parquet
```

若使用本地新增的 `configs/hydrostate_hpc.py`，它支持 `HYDROSTATE_DATA_ROOT`
和 `HYDROSTATE_MANIFEST` 环境变量；远端必须先获得该配置才可使用。若保持远端已有
`configs/hydrostate_v1.py`，可通过 `--cfg-options` 分别覆盖 train/val/test 的
`dataset.data_root`、`dataset.manifest`，无需修改数据文件。

同一版本的数据上恢复中断训练，用相同 work_dir 和 `resume=True`。增加样本后，
创建新的实验目录、指向新版本清单，使用 `load_from=上一阶段checkpoint` 与
`resume=False` 开始新的训练阶段，重新设置学习率日程。训练程序不会自动重新读取
正在增长的清单，因此不要覆盖正在训练的版本，也不要将更换数据集与断点恢复混为一谈。

扩展版本仍保持地理 split；验证/测试样本数也可能增长。比较两阶段模型时，
额外在同一份固定验证/测试清单上评估，避免因评估样本变化误判改进。

项目原有指南见 [HPC_FINETUNING.md](HPC_FINETUNING.md)。当前远端 PyTorch 为 2.7.1，
与本地项目声明的 2.9 系列不同；正式微调之前仍需核实运行环境并完成短训练检查。
