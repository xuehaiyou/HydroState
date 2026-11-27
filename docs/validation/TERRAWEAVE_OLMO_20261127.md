# 单 OlmoEarth TerraWeave 对照

在四级独立参数 TerraWeave 上仅保留 OlmoEarth 证据，移除 AnySat 编码器与对应输入投影。OlmoEarth v1.2 Base 完全冻结，目标时间查询、四个独立 URA、Cover/SM/ET 预测、产品可学习方差及人工监督沿用[双 FM 实验](TERRAWEAVE_20261126.md)。同一单 FM 下同时可与原 BDTA + MoTE 比较。

使用种子 3407、10,912 个训练窗、3,681 个开发验证窗、连续 6 轮、有效训练 batch 10、验证及测试 batch 8。按 manual_joint_score 选择一个检查点测试 11,168 窗。比较冰川三类、洪水、站点 SM/ET 和吞吐，跨卡速度须注明硬件。

[模型配置](../../configs/hydrostate_joint_terraweave_olmo.py)

11 月 27 日在 A100 80 GB PCIe 上启动（PID 160072），代码为 `d570f02`，运行目录为 `/mnt/storage/xiaozhen/HydroState/experiments/tw_olmo_d570f02/terraweave_olmo`。真实样本预检通过，micro-batch 5、累积 2，预检峰值张量显存 3.66 GiB。备用作业号为港大 4520191、地理系 40805。Slurm 候选各申请 20 GiB 主机内存，参考港大同协议单 FM 作业 4514558 的 batch 步骤峰值 RSS 16,760,072 KiB。
