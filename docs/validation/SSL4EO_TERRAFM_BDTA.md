# SSL4EO-L 与 TerraFM 的双 FM BDTA + MoTE 对照

研究问题是，在相同的三传感器输入、直接读取版 BDTA + MoTE 和监督预算下，由不同编码器分别读取 Landsat 与 Sentinel 能否改善站点和人工 Cover 精度。对照使用[原 OlmoEarth + AnySat 学习来源权重版](DIRECT_MULTIFM_BDTA_20261126.md)。

| 冻结编码器 | 输入与单位 | 初始特征及四级出口 |
|---|---|---|
| SSL4EO-L ResNet-50 MoCo | Landsat 8/9 TOA，B1–B11。按 30 m 聚合并还原官方字节缩放，九个反射波段取 0–0.4，两个热红外亮温波段取 248.15–318.15 K | stem、layer1–4 |
| TerraFM-Base | S1 VV/VH 的 dB 和 S2 L2A 的十二个波段按各自预训练统计归一化。项目 S1 使用 GEE GRD，原预训练使用 RTC，地形辐射条件可能存在差异 | patch、Transformer 第 3、6、9、12 层 |

两个 Sentinel 输入共用同一组 TerraFM 权重，分别前向。每个时间槽内，两个传感器在同一个 40 m 特征网格上按有效性取均值，仅一方有效时保留该方特征。两路 FM 的五级特征继续使用原解码器的目标时间查询、来源权重学习、双向适配、MoTE 及七个输出头。空间插值按原生特征中心定位；粗特征的空间支持不会因插值缩小。

权重分别来自 [TorchGeo](https://huggingface.co/torchgeo/ssl4eo_landsat/tree/fd1c4783667e1fd1c4ccc04cf04acc85c3f52af2) 和 [TerraFM 作者发布](https://huggingface.co/MBZUAI/TerraFM/tree/86f62795ff0e8dfae729e170e0ed82fb3e07ec9e)。TerraFM 实现和权重使用同一修订。

训练入口为 `hydrostate.experiments.dense_adaptation landsat_terrafm_bdta`。沿用原六轮协议、种子 3407、有效训练 batch 10、验证及测试 batch 8。训练、开发、测试分别为 10,912、3,681、11,168 个固定窗口。沿用原可学习产品方差和人工 Cover、站点损失，每轮验证，按开发集 manual_joint_score 选定一个检查点完整测试。报告冰川水体、冰雪、陆地的分项结果，洪水比例误差与 F1/IoU，以及站点 SM、ET 的 RMSE、Pearson R 和有效观测数。

[模型配置](../../configs/hydrostate_joint_landsat_terrafm_bdta.py) · [编码器及解码器接入](../../hydrostate/models/cosa/landsat_terrafm.py)
