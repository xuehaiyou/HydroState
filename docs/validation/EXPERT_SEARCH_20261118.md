# 完整专家架构与站点选模筛选（2026-11-18）

用户授权实现并在 HPC 运行：完整 MLoRE、多层 PLE、匹配 CGC、AdapterFusion、CoSA Router。
A/B/原 CGC 一同复验，用于统一有效 batch 与新的站点选模口径，共八组。

## 科学设置

- 沿用 architecture_search_20261117/selection_v2 和 stats_10：4528 训练窗口、2261 开发窗口。
- 各模型六轮、每两轮开发验证，FM 预训练初始化＋新增任务模块从头训练，不加载旧任务 checkpoint。
- 有效 batch=10，GPU 短测在其约数 1/2/5/10 中选择最大安全 micro-batch，梯度累积对齐。
  相对前轮有效 batch=5 有改变，故重新训练基准；不同 micro-batch 的 BatchNorm 统计仍有差异。
- 只按 `site_score = -(RMSE_SM/std_train_SM + RMSE_ET/std_train_ET)/2` 选模，两个目标必须均有观测。
  保留 Pearson R、RMSE、MAE、Bias 和产品一致性。产品不参与该分数。
- 短测 test API 显式绑定开发验证 loader/evaluator，仅作 checkpoint 重载检查；不读取封存测试集。
- 完整 MLoRE 使用官方 ViT-B 解码配置及负载均衡损失；详见[来源与差异](../../hydrostate/models/mlore/PROVENANCE.md)。
- PLE 的层级、门控和空间接口见 [PLE](../PLE.md)。原单层 CGC、单层 MLoRE 不覆盖。

## 验证与运行

本地已通过完整解码路径与固定官方 forward 的 FP32 数值对照、负载均衡公式、
PLE 共享门控解析结果和跨任务梯度、缺测掩膜、真实小型编码器的混合监督，以及
站点选模不受产品误差影响的测试。HPC 任务号和验收结果在提交后追加。

提交入口 `tools/experiments/expert_search.py`。每候选使用独立 GPU debug 作业探测、
前后向及 checkpoint 重载；通过验收后依据实际吞吐、历史短测到端到端的校准系数，
加 10% 时间余量提交训练。候选训练按 afterany 顺序依赖，某模型失败不阻止其他模型。
已有运行不会被中止或覆盖。
