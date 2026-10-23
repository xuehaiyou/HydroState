# 输入规格与独立标签包

当前基础影像格式仍为 `hydrostate_retrieval_5x8_native_v2`。新增标签使用独立目录、
独立版本和样本 ID 索引，既不改写基础影像/标签，也不要求重新下载影像。

## 规格的唯一入口

[`specs.py`](../hydrostate/data/retrieval/specs.py) 集中定义：

- `SOURCES`：产品集合及有效日期边界。
- `INPUT_SPECS`：波段顺序、产品分辨率、单位、QA 策略、合成方式、归一化和模型模态名。
- `LABEL_SPECS`：基础标签的来源波段、单位、尺度转换、物理范围、时间支持、监督类型和预测头激活。

输入分组、通道数、模型输入顺序和基础标签数组由规格派生；采集保存规格快照。
QA 和产品读取仍是实际代码，注册一个新产品名称不会自动实现其科学处理。
全新传感器还需要对应 QA/归一化适配器，并确认 OlmoEarth 的输入嵌入支持其通道。
目前输入时序为五个 8 天槽，静态或其他时序输入需要显式增加适配器。

扩展标签通过训练配置的 `label_specs` 声明，不修改全局基础七变量，
也不依赖程序启动顺序注册。默认七变量行为保持不变。

## 标签包结构

```text
labels/<product_version>/
  metadata.json          # 完整标签规格、来源、版本、索引哈希
  index.parquet          # sample_id → Zarr 分片和行号，含样本身份指纹
  data/part-00000.zarr    # 默认每片 256 个样本，逐样本压缩
  _SUCCESS.json          # 成功发布标记，绑定 metadata.json 哈希
```

关联验证包含 sample_id、划分、CRS、仿射变换、目标起止时间和输入版本。
基础 manifest 可换行序；不能把同 ID 的其他地点、日期或划分误接进来。
一个包可以只覆盖部分样本；其余样本的新标签全无效，基础监督仍可用。
同一扩展标签只能选一个包，当前不允许通过扩展包替换耦合的基础七标签。

| support | 写入 NPZ 的字段 | 训练监督 |
|---|---|---|
| `pixel` | `x[1,40,40]`、`x_valid`、`x_time[2]` | 30 m 像元 |
| `window_mean` | `x[1,1,1]`、`x_valid`、`x_time[2]` | 有效预测域的图块均值 |
| `native` | `x[K]`、`x_valid[K]`、`x_weights[K,40,40]`、`x_time[2]` | 预测按面积聚合到产品像元 |

数组值必须已经完成产品 QA、空间对齐和物理单位转换。`scale/offset` 记录产品转换约定；
导入器不会再次对准备好的物理量缩放。掩膜为 bool 或 0/1；有效值必须有限并符合声明范围。
每个有效原始产品像元的非负面积权重之和必须为 1。包读取时自动转换为模型的
`x_native_value/valid/weights` 命名并去掉无效尾部，拼批时补齐。

时间为 UTC epoch 秒，`target_interval` 必须与目标 PML 周期严格相同。
也可显式声明 `instant`（两个时间相同）或 `interval`（独立起止时间）；
有效监督时间必须位于该样本的 40 天输入窗口内，模型使用这个标签自己的时间查询。
这些检查验证时间声明，不会把瞬时观测自动变为 8 天平均值。

## 从已经准备好的标签创建一个包

下面只是格式示例，不代表已选定或采集 SWE 产品。把 `source`、`qa`、`aggregation`
等替换为实际产品定义，并生成经过处理的 NPZ。

`swe_spec.json`：

```json
{
  "version": "swe_v1",
  "source": "actual_product_and_processing_version",
  "label_specs": {
    "swe": {
      "source": "actual_product_id",
      "band": "SWE",
      "units": "mm",
      "support": "native",
      "resolution_m": 500,
      "qa": "describe_actual_quality_filter",
      "aggregation": "describe_actual_target_period_mean",
      "temporal": "target_interval",
      "activation": "softplus",
      "valid_min": 0,
      "capacity": 128
    }
  }
}
```

`records.parquet` 包含唯一的 `sample_id` 和 `npz_path` 两列，路径相对于 records 文件。
NPZ 存放上表中的字段，样本 ID 必须来自现有 `samples.parquet`。

```bash
python tools/data/build_label_pack.py \
  --manifest "$HYDROSTATE_DATA_ROOT/samples.parquet" \
  --records /path/to/records.parquet \
  --spec /path/to/swe_spec.json \
  --output /path/to/labels/swe_v1
```

也可在产品采集器中直接使用 `RetrievalLabelPackWriter.write(sample_id, values)` 流式写入，
省去中间 NPZ。写完才发布，不覆盖已有包；失败不发布半成品。

## 训练、审计与统计

```bash
export HYDROSTATE_LABEL_SPEC=/path/to/swe_spec.json
export HYDROSTATE_LABEL_PACK=/path/to/labels/swe_v1
export HYDROSTATE_TARGET_STATS=/path/to/stats_with_swe.json

python tools/validate_zarr.py configs/hydrostate_extended.py --pixel-limit 0
python tools/data/compute_target_statistics.py \
  --config configs/hydrostate_extended.py --output "$HYDROSTATE_TARGET_STATS" --num-workers 4
python tools/train.py configs/hydrostate_extended.py
```

`configs/hydrostate_extended.py` 读取同一份 JSON 规格。公共配置绑定过程自动添加任务、
预测头和数据读取声明。默认头宽度随 decoder，激活来自标签规格；可在配置里显式覆盖头。
审计检查基础数据及所选标签包。统计只使用训练划分和有效监督，并绑定规格、包元数据、
索引与不确定性筛选参数；切换标签包后必须重新生成统计。
路径为相对路径时，以基础数据根目录为参照。

准备成对 MTLoRA / LoRA 时，在 `tools/data/prepare_training.py prepare` 后加入
`--config configs/hydrostate_extended.py`。准备工具将相同标签选择写入两个展开配置。
基础集合是否完成仍按原七组配额检查；独立标签包不改变该集合的完成状态。
