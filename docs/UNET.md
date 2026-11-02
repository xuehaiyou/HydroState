# U-Net 随机初始化对照

`configs/hydrostate_unet.py` 提供不加载基础模型权重的监督学习对照。
模型类为 `UNetHydroStateModel`；所有参与计算的网络参数均训练。
它与 MTLoRA、shared-LoRA 和 Full 共用当前数据契约、划分、七任务监督及评估接口。
这是完整方法比较；网络结构和初始化同时变化，不能把差异单独归因于预训练。

## 输入与结构

复用 `RetrievalDataset` 和现有归一化，不修改已发布样本或重新估计输入统计。
S1/S2 的固定归一化常数来自现有数据契约，保留这些常数不等于加载 FM 权重。

按契约顺序拼接五个时间槽：125 个影像通道、120 个观测质量通道、30 个有效性通道，
加上目标日期的正弦/余弦季节编码，共 277 通道。通道数从 specs/contract 派生。
质量摘要仍为有效日期数及槽内平均/最早/最晚日期，均除以 8，与当前 FM 质量融合相同。
影像按完整传感器有效掩膜清除缺测值，质量摘要按原始分辨率组掩膜清除缺测值；
有效零值与缺测通过掩膜区分。季节编码按 UTC 日历年的实际长度计算，包含闰年。

- 编码通道：64 → 128 → 256 → 512；两层 3×3 卷积、GroupNorm、ReLU。
- 三次最大池化：120 → 60 → 30 → 15。
- 解码：双线性上采样、同尺度编码特征拼接、卷积，恢复到 120 × 120。
- 输出特征按 3×3 平均池化到 40 × 40，再接与 FM 相同的七个轻量预测头。
- 卷积块使用 Kaiming 随机初始化；归一化为标准初始化，预测头采用现有随机初始化。
  fresh run 默认 `load_from=None`、`resume=False`；续训需显式指定已有 U-Net checkpoint。

输入仍包含未来上下文，属于回溯反演。模型只预测中心目标窗口，所有任务必须同窗；
允许 PML 年末周期短于 8 天，拒绝异步任务查询或平移的目标窗口。
独立标签包只有目标窗口相同时才可接入，不能把这个限制解释为通用时间查询能力。

## 监督与评估

`RetrievalModelMixin` 在 U-Net 与 MTLoRA 间共享时间对齐检查、SM 有效域和总 ET 派生。
覆盖比例同组 softmax；SM 使用有效陆地面积加权、权重停止梯度，标签值不缩放；
PML 继续按原始像元支持范围监督，三个 ET 分支非负。推理 SM 有效域使用预测覆盖。
输出分辨率仍为 30 m；对输入的重采样不产生新的观测分辨率。

默认沿用有效 batch=16（microbatch=2、累积=8）、50 epochs、预热/余弦调度、
任务权重和 `hydro_score` checkpoint 选择。AdamW 起始学习率 3e-4、weight decay=0.01，
没有 FM 分层学习率衰减；这些是起始配方，不代表已完成调参。
微调 microbatch 时保持有效 batch，并重新核对预热对应的优化器更新数。

正式比较固定清单、空间划分、训练统计、增强和样本暴露次数，采用可比的验证调参预算。
建议至少三个随机种子，报告各任务精度、有效监督量、参数量、耗时、吞吐和峰值显存。
监督评估保持观测定义的支持范围，另外报告预测有效覆盖率。
若 U-Net 在固定预算内尚未收敛，另报延长训练的结果和新增成本。

## 使用

设置与其他组相同的数据和训练统计，并指定独立工作目录：

```bash
export HYDROSTATE_WORK_DIR=/path/to/work_dirs/retrieval_unet
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/hydrostate_unet.py
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/hydrostate_unet.py CHECKPOINT.pth
```

无需 `HYDROSTATE_MODEL_PATH`。本地包装支持
`HYDROSTATE_CONFIG=configs/hydrostate_unet.py bash scripts/train.sh`；
Slurm 包装支持 `scripts/slurm/train_retrieval.sh unet`，保留发布校验和单 GPU 要求。
正式作业时间应依据 debug 测量加 10% 余量显式覆盖，不能直接沿用历史 96 小时声明。

区域推理与原有导出、拼接格式一致：

```bash
python tools/predict_region.py \
  --root /path/to/prepared_region \
  --model-config configs/hydrostate_unet.py \
  --checkpoint /path/to/unet_checkpoint.pth \
  --statistics /path/to/training/target_stats.json \
  --device cuda --batch-size 2 --workers 0
```

U-Net 不需要 `--pretrained`；FM 可通过该参数覆盖配置中的本地权重路径，
仍要求非空 `config.json` 和 `weights.pth`。区域 benchmark 同样支持省略该参数。
模型配置必须对应 checkpoint，严格加载参数。

当 debug 队列需要等待时，可用 `tools/experiments/submit_after_debug.py --run-root RUN`
在 `afterok` CPU 作业中接续提交。运行记录必须包含冻结源码/环境、debug 提交命令、
GPU 容量和实查的常规队列限制。工具要求完整一轮训练/验证、有限损失和至少 10% 显存
余量；按实测完整轮次耗时乘正式轮数再加 10% 申请时限，超限时按 epoch 拆分续训。
正式训练重新随机初始化，不读取 debug 权重；独占意向文件和逐次回执防止重复提交。

本地验证范围与硬件限制见 [验收记录](validation/UNET_20261102.md)。
