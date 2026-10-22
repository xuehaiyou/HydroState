# 当前数据契约

可执行定义位于 [`contract.py`](../hydrostate/data/retrieval/contract.py)，
版本为 `hydrostate_retrieval_5x8_v1`。训练配置从这里导入 `TASKS`、`PROFILE`、
`VERSION` 和 `OUTPUT_SCALE`，不再复制任务字典。

- 输入：5 个时间槽，120×120、10 m；S1/S2/Landsat 分别为 2/12/11 通道。
- 每传感器保存独立有效掩膜、有效日期数和平均/最早/最晚日期偏移。
- 输出：40×40、30 m；覆盖三分量、陆地 SM 和水/冰雪/陆地 ET 共七个头。
- SM 保存粗产品标量及独立监督域；ET 保存原始像元值、有效掩膜和面积权重。
- `_valid` 和质量摘要定义见 [完整规范](RETRIEVAL_PIPELINE.md)，不以填充值判断有效性。

每个成功样本写入一对不可变输入/标签 Zarr 分片。`samples.parquet` 记录分片位置、
版本、中心频率组、地理划分、目标日期等。`collection.json` 固定参数与生产代码哈希；
逐样本提交记录用于续采。七组配额全部完成后才发布 `_COLLECTION_SUCCESS.json`。
改变参数、生产代码或数据定义时使用新目录，不将新旧样本混写。

`RetrievalDataset` 验证版本、输入配置、任务和分片属性；`audit.py` 检查全部分片契约，
并按 `--pixel-limit` 抽查或全查数组。统计量只从训练划分生成，绑定 manifest 和数据契约；
训练前重新检查指纹。

`storage/schema.py` 只定义公共 `ArraySpec` / `ShardSchema`，不提供隐式旧格式。
历史格式见 [归档 schema](history/DATA_SCHEMA_BEFORE_RETRIEVAL.md)；正式入口不自动兼容。
