# 单 FM BDTA + MoTE 的产品 NLL 方差对照

比较已完成的[单 OlmoEarth BDTA + MoTE](BDTA_MOTE_20261125.md)与同架构重新训练的固定方差组。唯一的训练干预是将七个产品输出的标准化高斯方差固定为 1。固定组按原式计算 ½ 标准化平方残差并保留高斯常数、任务系数、观测权重及缺测掩膜。站点 SM、站点 ET 的标准化 MSE 与人工 Cover 比例 MSE 各自沿用系数 2。

两组从相同 OlmoEarth 预训练权重出发。固定组新建输出目录，种子 3407，训练 10,912 窗，逐轮开发验证 3,681 窗，共 6 轮。训练有效 batch 10，基线为每批 5 窗、累积两次；验证及测试 batch 8。按开发 manual_joint_score 选优后，对 11,168 个独立测试窗口报告冰川三类分项、洪水水体比例与分类，以及站点 SM、ET 的 RMSE 和 Pearson R。产品一致性单列。

[实验配置](../../configs/hydrostate_joint_bdta_unit_variance.py) · [训练入口](../../hydrostate/experiments/dense_adaptation.py) · [固定与可学习方差的损失及梯度测试](../../tests/test_uncertainty_nll.py)
