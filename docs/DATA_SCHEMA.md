# 当前数据契约

可执行定义位于 [`contract.py`](../hydrostate/data/retrieval/contract.py)，
版本为 `hydrostate_retrieval_5x8_native_v2`。训练配置从这里导入 `TASKS`、`PROFILE`、
`VERSION` 和 `OUTPUT_SCALE`，不再复制任务字典。

- 存储：5 个时间槽，1.2 km 同范围图块；各波段保留产品发布分辨率，见下表。
- 模型输入：读取时统一为 120×120、10 m；S1/S2/Landsat 分别为 2/12/11 通道。
- 每个分辨率组保存独立有效掩膜、有效日期数和平均/最早/最晚日期偏移。
- 输出：40×40、30 m；覆盖三分量、陆地 SM 和水/冰雪/陆地 ET 共七个头。
- SM 保存粗产品标量及独立监督域；ET 保存原始像元值、有效掩膜和面积权重。
- `_valid` 和质量摘要定义见 [完整规范](RETRIEVAL_PIPELINE.md)，不以填充值判断有效性。

| 存储组 | 波段 | 分辨率 | 每槽空间尺寸 |
|---|---|---|---|
| `s1_10m` | VV, VH | 10 m | 120×120 |
| `s2_10m` | B2, B3, B4, B8 | 10 m | 120×120 |
| `s2_20m` | B5, B6, B7, B8A, B11, B12 | 20 m | 60×60 |
| `s2_60m` | B1, B9 | 60 m | 20×20 |
| `landsat_15m` | B8 | 15 m | 80×80 |
| `landsat_30m` | B1–B7, B9–B11 | 30 m | 40×40 |

每组包含 `组名`、`组名_valid`、`组名_quality`，分别为
`(5,C,H,W)` float16、`(5,1,H,W)` uint8 和 `(5,4,H,W)` float16。
影像保存归一化后的 QA 有效日期均值；不是逐景原始 DN。
`input_grids` 属性保存每组 CRS、仿射变换、波段顺序和尺寸。
各组共享图块外边界与 CRS，保留产品像元大小；不声称保留每幅原始影像的网格原点。
Landsat B10/B11 按产品发布的 30 m 保存，不能还原传感器的 100 m 原始观测。
波段分辨率以 [GEE Landsat L1](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1)
和 [GEE Sentinel-2 SR](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
官方目录为准。

`RetrievalSchema.inputs` 定义磁盘数组，`model_inputs` 定义读取后的数组；
`alignment.py` 在 CPU 上按像元中心最近邻对齐，15→10 m 同样按空间坐标确定索引。
光谱、有效性和日期使用相同索引，不插值观测次数。
模型波段按预训练顺序重组，每种传感器的有效掩膜取其各组掩膜的交集，
交集之外光谱置零。各组独立质量摘要仍保留，并由 `CompositeQualityFusion` 接收。
增强在完成 10 m 对齐后统一执行，30 m 标签同步翻转。

每个成功样本写入一对不可变输入/标签 Zarr 分片。`samples.parquet` 记录分片位置、
版本、中心频率组、地理划分、目标日期等。`collection.json` 固定参数与生产代码哈希；
逐样本提交记录用于续采。七组配额全部完成后才发布 `_COLLECTION_SUCCESS.json`。
改变参数、生产代码或数据定义时使用新目录，不将新旧样本混写。

`RetrievalDataset` 验证版本、输入配置、任务和分片属性；`audit.py` 检查全部分片契约，
并按 `--pixel-limit` 抽查或全查数组。统计量只从训练划分生成，绑定 manifest 和数据契约；
训练前重新检查指纹。

`storage/schema.py` 只定义公共 `ArraySpec` / `ShardSchema`，不提供隐式旧格式。
历史格式见 [归档 schema](history/DATA_SCHEMA_BEFORE_RETRIEVAL.md)；正式入口不自动兼容。
