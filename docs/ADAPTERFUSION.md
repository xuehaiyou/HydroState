# 实验 D：冻结 shared-LoRA 专家与 AdapterFusion

配置：`configs/hydrostate_joint_adapterfusion.py`。

以产品＋站点联合训练的 shared-LoRA 固定最佳快照初始化，严格加载所有旧权重，
再新增每个任务的 AdapterFusion 和零初始化标量 gamma。七个独立时间查询、三深度融合、
原有空间解码和物理输出定义保持不变。编码器、LoRA、patch embedding、LayerNorm、
任务分支、输出头和原有损失权重全部冻结，并保持 eval 模式。
只有融合与 gamma 更新；冻结解码器仍保留输入梯度。

官方来源：<https://github.com/adapter-hub/adapters>，固定 commit
`53a1ea164c07498be7d522aed96c9e600aa3b38b`，
`src/adapters/methods/modeling.py` 中的 `BertFusion` 与 `Adapter.init_bert_weights`。
提取代码保存在 `hydrostate/models/adapterfusion/vendor/`，保留 Apache-2.0 LICENSE。
模块体不变，仅将类型注解依赖简化为 dict，并保留所需初始化 helper；不引入整个 NLP 框架。
使用 Q/K/V、近单位 V 初始化、固定温度 1、dropout=0、softmax 前 value 投影。
这属于任务特征上的领域适配，不是完整复现论文中的逐层 Adapter 网络。

空间融合放在上采样前：`F_i + gamma_i * delta_i`。官方 residual 输入为零，避免重复加基线。
每个接收任务包含自身和所有来源专家；按接收任务目标时段重新查询来源专家。
相同时间窗口在单次前向内复用，跨前向不缓存。
标签有效性不参与路由；观测有效性在专家注意力之前筛选。
官方接口没有 mask：按来源可用性打包 token，移除无效专家后调用原始模块。
没有有效来源的 token 增量为零；保持基线原有无观测输出行为，不额外改变原输出掩膜。

训练继续使用固定数据、统计量、无放回采样、站点损失系数 2、站点选模权重 2/3。
融合训练最多 10 epoch，AdamW lr=1e-4、100 step warmup、cosine decay；不增加专家预训练。
GPU debug 确定 batch，梯度累积 1；新阶段有效 batch 与原基线的差异明确记录。
单卡、最多 6 CPU，walltime 基于实测训练＋验证＋启动成本加 10%。

运行必需环境变量：`HYDROSTATE_BASELINE_CHECKPOINT`、`HYDROSTATE_DATA_ROOT`、
`HYDROSTATE_TARGET_STATS`、`HYDROSTATE_SITE_SPLIT_ROOT`、`HYDROSTATE_MODEL_PATH`。
快照及校验和放在运行目录，不修改旧任务的 checkpoint。

验收：`python -m pytest tests/test_adapterfusion.py tests/test_mtlora.py`；
真实数据 `python tools/experiments/validate_adapterfusion.py --output ...`；
显存与吞吐复用 `tools/benchmarks/joint_training.py --config ... --batch-size ... --output ...`。
验证覆盖严格 checkpoint、零残差回退、关闭后回退、不同目标窗口、无效/空来源和梯度隔离。

D 与其同一 A 快照直接比较；与 B/C 比较需标注额外融合训练预算。
分别报告完整验证集和产品—站点共同子集，禁止混用旧 epoch 3 的子集结果。
正式结果按同一个综合最佳 checkpoint 报告，不拼接不同轮次各变量最优。
新增变量适配尚未实现/验证，不以当前任务融合收益代替迁移证据。
