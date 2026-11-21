# 人工覆盖点外部验证

## 范围与支持

首版读取 `glacier-snow-cover-mapping/inputs-outputs/models_training_data/Sentinel-2_TOA_training_data.csv`。
原始五分类中的雪、阴影雪、冰映射为 snow_ice；岩石/碎屑为 land；水为 water。
SR 与 TOA 表是同一组点，不重复评分。原项目训练标签只作为本项目外部测试；
仍需逐 checkpoint 检查其训练区域，不能据文件名声明独立性。
官方雪/非雪验证表不能用于完整三类评估，非雪包含裸冰。

数据根目录：`/scr/u/xiaoz/HydroState/samples/glacier_cover_v1`。
`collection.json` 固定源文件、生产代码、类别、输入契约与筛选规则。
`points.parquet` 保存原始行、坐标、日期、类别、唯一 point_index、窗口及输出行列。
`plan.parquet` 保存全部窗口；`samples.parquet` 只保存成功窗口，均为 `split=test`。
每个 `zarr/<sample_id>/inputs.zarr`、`labels.zarr` 使用公共 SampleShardWriter 成对提交。
输入保持 retrieval_5x8_native_v2 的原生波段网格、五槽 QA 合成、归一化与质量数组。
标签契约为 hydrostate_cover_points_v1，变长 K 点数组；不生成面标签、不计算覆盖比例 RMSE。

900 m 固定所有权网格加 150 m 四周上下文，得到 1200 m 输入窗口；每点只归属一个窗口。
日期对应以 1 月 1 日为锚点的 8 天周期（与现有站点日历一致），年末截断目标周期。
目标仍为周期状态；原始人工标签只有日期精度，记录日期偏移，不伪造单日反演。

## 入口

使用项目 Python 3.12 环境，从 Git 固定版本执行并设置 `PYTHONPATH=.`。

```bash
python tools/data/cover_points.py prepare --source SOURCE.csv --root DATA_ROOT
python tools/data/cover_points.py collect --root DATA_ROOT --project GEE_PROJECT --pilot
python tools/data/cover_points.py audit --root DATA_ROOT --pilot
python tools/data/cover_points.py collect --root DATA_ROOT --project GEE_PROJECT
python tools/data/cover_points.py audit --root DATA_ROOT --publish
```

正式采集要求代表性 pilot 审计通过。输入不足单独记录 rejected，暂时网络异常向外报错，
重试同一命令会跳过已提交和已明确拒绝的窗口。采集使用一层有界线程池，默认 4 并发。
冻结生产源码或标签变化时要求新数据目录，不能覆盖成功分片。
标签来源有效性与模型输入支持独立：同窗部分点缺测不删除其人工标签。
发布记录保存清单和来源身份；模型读取检查发布文件一致性。

```bash
python tools/validation/evaluate_cover_points.py predict \
  --root DATA_ROOT --config FROZEN_EXPERIMENT_CONFIG.py \
  --checkpoint EXISTING_SELECTED_CHECKPOINT.pth --output NEW_RESULT_DIRECTORY \
  --batch-size 4
python tools/validation/evaluate_cover_points.py compare \
  --predictions A=RESULT_A --predictions B=RESULT_B --output common_metrics.json
```

`--limit N` 为短测，结果明确标记 smoke，不能用于正式模型比较。
模型由原配置注册类构建、严格加载 state_dict，复用其训练统计量，不改变原选模。
输入只进入无标签预测分支；人工标签不参与 DW/SMAP/PML 监督聚合。
各架构使用原始已选 checkpoint，不以新增 Cover 分数重新挑选 checkpoint。

## 独立性和报告

对配置实际选中的训练窗口（含站点）建立地理圆索引，检查测试窗口外接圆加 35 km
保守余量，跨日期也排除重叠。该余量保守覆盖当前粗支持近邻要求，不表示精确足迹。
保存逐窗口训练重叠；审计绑定实际训练配置，而非当前 ACTIVE 指针的猜测。
历史配置若已无法恢复原训练发布版，应先定位原清单，不放宽审计。

结果包含 predictions.parquet、training_overlap.parquet、provenance.json、metrics.json。
保留输入拒绝窗口的点在总分母报告中。仅有效且不重叠的点参与独立评分。
每类 Precision/Recall/F1/IoU、混淆矩阵、地点和地点日期分组与 Macro-F1 均明确样本量。
无定义指标为 null，Macro-F1 同时报参与类别数。同像元异类标注单列冲突数量；
补充一致标签的唯一像元指标，不作多数投票。跨模型比较使用共同有效且独立的点交集。
不把岩石/碎屑环境结果概括为一般陆地精度，不把一万多个邻近点视为独立场景。
模型输出为表面比例，argmax 表示主导表面分类，点位结果不能替代面积比例验证。

## 验证

`python -m pytest tests/test_cover_points.py` 覆盖标签映射、重复冲突、真实点-网格
重投影、Zarr 往返、断点跳过、pilot 发布门槛、缺测与空类别分类统计。
真实采集、GPU 启动及各模型结果另记 `docs/validation/`，不以单元测试代替实采验收。
