# 产品直接对站点：共同验证交集

新增独立评估流程，不改训练标签、划分、统计量、选模分数或正在运行的任务。
配置/分片清单绑定指纹；观测键由 cohort、sample_id、变量和观测序号组成，避免合并同址传感器。
仅处理固定 validation 划分；test 留待最终评估，不按产品误差选取或排除样本。

- SMAP：复用 `comparison.smap_observations/pair_sm`，原发布网格、上午 QA=0，
  仅在共同合格上午日期上计算 SMAP 和站点均值。保存原训练均值与重新配对均值。
  所有模型对同一重新配对站点值评分；模型仍是 8 天状态估计，因此共同日期只是窗口内
  的观测估计，不称为逐日预测或真实过境时刻匹配。粗足迹与点位代表性差异仍存在。
- PML：只使用 ET，不与 Ew 求和。把现有站点 30 m 权重映射到原生 PML 像元，
  按相同权重聚合，采用网格中心积分近似，不插值出新的高分辨率观测。
  任一有贡献原像元缺失、负值或高置信度 DW 陆地面积不超过产品配置门槛，则整条不配对；
  不对缺失支持重新归一化，不把未知当陆地。覆盖规则直接读取现有产品采样配置。
  使用已发布的波文比校正站点 ET。站点当地日和 PML 名义 UTC 周期偏移显式保留，
  无逐时产品不能声称已经消除该偏移。PML 原生像元内部未解析结构仍是限制。
- 公平交集：产品有效 AND B有效 AND C有效 AND shared-LoRA有效。
  产品和模型均对同一站点值计算 R²=1−SSE/SST、RMSE、MAE、Bias；另外报告空间组等权
  RMSE/MAE/Bias、窗口/观测数和空间组数。全量原验证指标保留，不能直接横比交集指标。
- 模型权重先复制固定快照并记录 SHA256；评估不追随训练中变化的最佳文件。
  对照优先选择三个模型共同已完成的同一 epoch，并标明其不是最终最佳模型比较。

入口：

```bash
python tools/data/collect_site_product_baseline.py \
  --release "$PINNED_SITE_RELEASE" --output "$BASELINE_ROOT" \
  --project cropland-477906 --workers 4 --limit 2
# 小试通过后去掉 --limit，复用相同 rows/cache 续跑；不用重复下载影像。
python tools/experiments/evaluate_site_product_baseline.py predict \
  --config "$CONFIG" --checkpoint "$FROZEN_CHECKPOINT" \
  --pairs "$BASELINE_ROOT/pairs.jsonl" --output "$PREDICTIONS" --batch-size 4
python tools/experiments/evaluate_site_product_baseline.py score \
  --pairs "$BASELINE_ROOT/pairs.jsonl" \
  --prediction B=b.json --prediction C=c.json --prediction shared=shared.json \
  --output comparison.json
```

`plan.json` 固定输入与质量规则，`rows/` 为可续跑观测结果，`cache/` 保存产品响应及原生支持质量，
`pairs.jsonl` 为完整配对记录，`errors.json` 为可重试查询错误，`collection.json` 报告排除原因。
网络错误与真实产品缺失分开；有查询错误时返回失败，不能宣称全量完成。
下载采用最多 4 个线程，无嵌套工作池；同一缓存键有文件锁，重复点位与多传感器复用查询。
不下载新的 S1/S2/Landsat 输入，不改变已发布样本。
