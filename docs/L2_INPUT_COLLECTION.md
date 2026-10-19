# GEE 样本补充与 HPC 本地输入采集

两条数据来源共用输入语义：GEE 用于小样本下载，HPC 本地影像用于区域推理准备。
区域准备有三个后端：`pixels`、`drive`、`local`。新后端无需联网或 Google 凭据。

## 数据合同

`landsat_l2_sr_v1` 保留现有 S1/S2 输入定义，新增七通道 Landsat 地表反射率：

| 数据 | 输入语义与质量规则 |
|---|---|
| Landsat | LC08/LC09 C2 T1 L2，`SR_B1`–`SR_B7`；`DN * 0.0000275 - 0.2`；float32 物理反射率；保留负反射率；排除 QA_PIXEL bits 0–5、QA_RADSAT 非零和填充值；保留水体 |
| S2 | L2A；按每份 SAFE 元数据的 BOA_ADD_OFFSET 和量化系数转换成 GEE SR Harmonized 的非负 10000 倍反射率；然后沿用现有 OlmoEarth 归一化；SCL 排除 0/1/3/8/9/10 |
| S1 | 双极化 IW GRDH Sigma0 dB；保留 nodata 掩膜；沿用现有 OlmoEarth 归一化；同一窗口序列维持相对轨道一致 |

三者按最近邻对齐现有 10 m 输入网格，保留真实观测时间。重采样不增加 Landsat
原生 30 m 的空间信息。S2 保留原有十二波段顺序。`ST_B10` 本次不作为输入收集，
因此温度缺失不会剔除有效反射率，也不会引入同源 LST 标签泄漏。

本地 S1 的 Sigma0/dB 定义可核验，但其预处理与 GEE 的数值等价性尚不能仅凭
文件名确认；S2 不同重处理版本也可能存在差异。区域报告保留此验证状态。
正式训练/推理前仍需相同观测的跨来源像元比较。

官方产品定义：
[Landsat C2 L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)、
[S2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)。

## 给已有样本补 Landsat L2

```bash
# 默认 merged_v1 全部 20,000 样本，四个工作线程。
bash scripts/run_gee_landsat_l2.sh

# 代表性试采。与全量共用断点缓存，但发布到单独的包。
OUTPUT_DIR=/mnt/d/Hydrostate/samples/merged_v1/inputs/landsat_l2_sr_pilot_v1 \
  bash scripts/run_gee_landsat_l2.sh --limit 16
```

严格匹配原 manifest 中选中的 Landsat L1 场景到同次观测的 L2，不重新选时间。
原槽位为空时保持为空；L2 缺失时记录缺失，网络错误记录为失败并可重试。
失败的请求不会发布成“无有效像元”。样本 ID、划分、位置、标签和旧 L1 输入不改。

输出：`merged_v1/inputs/landsat_l2_sr_v1/`，使用现有 VariablePack 格式：

- `metadata.json` / `index.parquet` / `_SUCCESS.json` / `coverage.json`。
- `data/part-*.zarr` 是普通目录式 Zarr，不是 zip 或归档格式。
- `landsat_l2`: `(N,4,7,120,120)` float32 反射率。
- `landsat_l2_valid`: 同形状布尔掩膜。
- `landsat_l2_time`: `(N,4)` UTC epoch 秒。

`.landsat_l2_sr_v1.work/checkpoint.sqlite3` 是可恢复的临时下载缓存，`run.log` 记录进度。
试采成功的数据可在全量采集中复用。完成发布前不会暴露半成品包。
训练数据集可通过 `input_packs=[dict(path="inputs/landsat_l2_sr_v1",
variables=["landsat_l2"])]` 读取新增变量；此设置本身不会替换模型的旧 Landsat 分支。
新模型必须显式配置 L2 输入适配、停用旧 L1 分支并重新训练/微调。

## HPC 本地区域输入

```bash
# 复用已有鄱阳湖范围、空间网格和训练重叠记录，不请求 GEE。
python tools/predict_region.py init \
  --root /scr/u/xiaoz/HydroState/inference/poyang_local_l2_201704_202512 \
  --config configs/inference/poyang_local_l2.json \
  --template-root /scr/u/xiaoz/HydroState/inference/poyang_201704_202512

python tools/data/prepare_region_inputs.py \
  --root /scr/u/xiaoz/HydroState/inference/poyang_local_l2_201704_202512 \
  --backend local --local-config configs/data/hpc_local_l2.json --workers 4
```

`--dates 2020-07-15` 可只准备一个已有计划日期，省略则处理整个计划。
`scripts/slurm/32_prepare_region_local.sh` 可在无网络 CPU 节点运行；提交前设置冻结的
`HYDROSTATE_CODE_ROOT` 与 `HYDROSTATE_SOURCE_HASHES`。不要在主登录节点做大规模解码。

本地流程：建立源文件索引 → 按现有四个 16 天槽位读取块掩膜 → 筛选窗口 →
只处理被选中的完整波段块 → 校验并发布日期 manifest。相同源块跨窗口/日期复用。
实际路径、采集时间、S2 元数据、源文件大小和修改时间保存在 `local_catalog.json`。
Landsat 在打开波段之前检查 MTL：T2 产品记录为 `landsat_tier_2` 并跳过，继续处理
T1；缺失或未知的质量等级、缺文件、缺质量层、缺偏移元数据或源文件变化仍明确失败。

`local_catalog.sqlite` 逐景提交索引断点，同时保存每个传感器已发现的文件列表。
重启时复用文件列表和已完成的元数据检查，只重试未成功的场景；复用前核对源文件
大小和修改时间。所有场景检查结束后才发布完整的 `local_catalog.json`，其中
`excluded` 保存被排除的 T2 路径与原因。`local_catalog_progress.json` 记录当前扫描
阶段、完成量、复用量、排除量和出错场景。文件列表冻结后，新增源影像需使用新的区域目录。

输出结构：

```text
poyang_local_l2_201704_202512/
  plan.json, local_catalog.json, local_contract.json
  local_catalog.sqlite, local_catalog_progress.json
  preparation.sqlite
  quality_cache/{s1,s2,landsat}/*.zarr
  input_cache/{s1,s2,landsat}/*.zarr
  selections/YYYY-MM-DD.parquet
  manifests/YYYY-MM-DD.parquet
  reports/local_YYYY-MM-DD.json
```

索引限于配置里的鄱阳湖六个 S2/S1 瓦片和四个 Landsat 轨道行号，避免扫描整个数据池。
S1 裁成多个瓦片时分别保留，不能按原始场景 ID 去重丢掉其他空间块。
当前本地数据不覆盖全部 2017–2025 年，报告记录缺失传感器和有效窗口数。
本地后端不会暗中联网补缺；GEE 补缺需明确使用同一 L2 数据合同。

旧 `pixels`/`drive` 区域通道与原 L1 缓存保持原合同；它们目前不负责新 L2 区域补缺。
新区域根目录不会覆盖旧下载成果。现有三变量 L1 checkpoint 无法直接消费 L2，
`predict` 会明确拒绝，以免产生看似成功但输入不匹配的结果。本次交付的是输入采集，
并未提交新训练或区域预测任务。

## 代码位置

- `hydrostate/data/input_products.py`：输入版本、波段、单位与转换。
- `hydrostate/data/gee/workflows/landsat_l2.py`：GEE 同次观测 L2 样本生产。
- `tools/data/extend_gee_landsat_l2.py`：断点下载与普通 Zarr 输入包发布。
- `hydrostate/data/local/catalog.py`：本地文件发现、元数据与索引。
- `hydrostate/data/local/index.py`：文件列表与逐景索引的事务式断点保存。
- `hydrostate/data/local/reader.py`：块读取、投影、掩膜、S2 偏移校正。
- `hydrostate/data/local/region.py`：区域准备、完整性校验与发布。
- `hydrostate/data/local/planning.py`：离线复用区域计划。

验证：`python -m pytest -q tests/data/test_l2_inputs.py tests/data/gee tests/test_regional_inference.py`。
