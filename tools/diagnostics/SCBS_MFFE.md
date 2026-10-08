# SCB-S / D-FINE-S：MFFE 独立结构实验

## 目标和证据边界

五次基线总体 AP 为 56.04 ± 0.11，write AP 为 49.50 ± 0.27。
run1 中 write→read 为 210 例，read→write 为 156 例。writing 通常已有
空间上合理的候选框，但正确分类、评分和最终筛选仍有不足。

中尺度 AP 较低只是观察结果，不能直接证明 Encoder 丢失了判别信息。
本实验检验“在多尺度融合阶段增强局部细节能否改善行为区分”，不是
宣称已经定位原因，也不保证 AP 提升。MFFE 是本项目的实验实现，
论文创新性仍需另行检索和验证。

## 插入位置

HGNetv2 → 通道投影 → P5 AIFI → 自顶向下 FPN → **P3/P4 MFFE** →
底向上 PAN → 原有三尺度 Decoder。

- P3/P4 步长仍为 8/16，不增加 P2，不增加 query。
- 局部细节来自同尺度的融合前特征，语义引导来自 FPN 输出。
- 放在 FPN 后：此时已有较强语义，可用它选择细节分支。
- 放在 PAN 前：增强结果还能经过原有 PAN 传向较粗尺度，不只是修改
  最后一个输出张量。
- P3/P4 不是 COCO 的 medium 标签。模块同样可能影响其他尺度；训练和
  推理都不使用 GT 尺度来决定是否启用模块。

## 公式（l ∈ {P3, P4}）

设 X_l 是通道投影后的融合前特征，T_l 是同尺度 FPN 输出。

\[
D_l = \operatorname{SiLU}(\operatorname{GN}(W_d X_l)),\qquad
S_l = \operatorname{SiLU}(\operatorname{GN}(W_s T_l))
\]

三个分支以深度卷积提取不同感受野的细节：

\[
B_{l,k}=\operatorname{SiLU}(\operatorname{GN}(\operatorname{DWConv}_{k}(D_l))),
\quad k\in\{3,5,7\}
\]

融合前细节和融合后语义共同决定每个位置的分支权重：

\[
\pi_l=\operatorname{softmax}_{\text{branch}}(W_g[D_l;S_l]),\qquad
H_l=\sum_{k\in\{3,5,7\}}\pi_{l,k}\odot B_{l,k}
\]

最终保留原 FPN 主路径，增加一个幅度受限的残差：

\[
\widetilde T_l=T_l+\alpha_l\operatorname{GN}(W_oH_l),\qquad
\alpha_l=0.5\tanh(a_l)
\]

初始 alpha=0.1，因此 a_l=atanh(0.2)。训练后 alpha 可学习为正或负，
幅度限制在 0.5 内。两个尺度独立学习，不复用之前 Highres Residual
实验的 alpha。使用 GroupNorm 避免新增分支依赖 BN 的运行统计；门控
softmax 在 float32 中计算，再转换回当前特征精度。

门控权重采用小随机初始化、最终投影非零初始化，确保首步不阻断
细节分支和语义分支梯度。仅新参数新增随机初始化，原有 Encoder、
Decoder 参数初始化所用的 CPU 随机序列保持不变。

## 配置与不变项

训练：`configs/dfine/dfine_s_scbs_hrw_mffe.yml`。
诊断：`configs/dfine/dfine_s_scbs_hrw_mffe_diagnostics.yml`。

训练配置只继承 `dfine_s_scbs_hrw.yml` 并开启 MFFE：

- 三类及类别顺序不变：0=hand-raising、1=read、2=write。
- 数据集、划分、640 输入、batch=32 不变；当前与新基线统一为 110 epochs、epoch 100 停增强。
- Backbone、Decoder 结构、300 queries、VFL/FDR/LQE 和损失权重不变。
- 不叠加 Pairwise CE、margin、RW specialist、RFAConv 或其他旧模块。
- 修改共享 Encoder 特征可能同时影响分类和定位，**不是纯分类分支实验**。
- 关闭 `use_mffe` 不注册任何新增参数，并恢复原 Encoder 前向。

### 4090 显存修复：仅重计算新模块

2026-10-06 首次服务器运行在第 0 轮 backward 出现 CUDA OOM：当时空闲
仅 280.50 MiB，下一次分配需要 314 MiB。不是预训练加载或配置路径错误。
参数量小不等于训练激活占用小；P3/P4 上的大特征图和并行分支仍会增加
反向传播保存的张量。现有多尺度训练最高是 800，而不是始终 640。

当前 `mffe_checkpoint: true` 使用 PyTorch 的 non-reentrant activation
checkpoint，仅在训练且启用梯度时重计算 MFFE 分支。模型公式、参数、
batch=32、训练尺度和优化计划不变；代价是额外计算。MFFE 使用无运行
统计的 GroupNorm，**不**重计算原 FPN/PAN 的 BatchNorm，以避免重复更新
运行统计。验证、部署和无梯度推理不使用 checkpoint。

参考：<https://docs.pytorch.org/docs/2.9/checkpoint.html>。

CPU bfloat16 检查中，前向结果、输入梯度和参数梯度与不重计算版本一致；
模块级示例保存张量由 496,760 bytes 降到 196,608 bytes。这个数字不是
整模型 GPU 峰值。是否满足 4090 batch=32，必须在服务器检查。

新增参数 112,008；按现有日志的 deploy/profiler 口径，参数由
10,178,201 增加到 10,290,209。训练态参数量与 deploy 口径不同，不要混用。

## 同步与服务器训练

在本地终端同步这次需要的源码和配置，不使用 --delete：

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw.yml \
  src/zoo/dfine/hybrid_encoder.py \
  src/zoo/dfine/medium_finegrained.py \
  configs/dfine/dfine_s_scbs_hrw_mffe.yml \
  configs/dfine/dfine_s_scbs_hrw_mffe_diagnostics.yml \
  tools/diagnostics/test_medium_finegrained.py \
  tools/diagnostics/check_mffe_cuda_memory.py \
  tools/diagnostics/SCBS_MFFE.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器先做无数据、无下载的检查：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/test_medium_finegrained.py
```

服务器再做 CUDA 显存预检（读取实际训练数据，随机权重，两次优化步，
不保存权重、不更改已有输出）：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
python tools/diagnostics/check_mffe_cuda_memory.py \
  --config configs/dfine/dfine_s_scbs_hrw_mffe.yml
```

它分别使用配置的 batch=32、640 和最高训练尺度 800，并包含 AMP
backward、AdamW 状态和 EMA。预检的随机权重模型使用较小初始 loss scale
1024 来降低溢出误报；正式训练仍用原配置。仅这些抽样批次通过，不保证所有更密集
GT 的批次或其他进程抢占显存时都能通过。若仍然 OOM，请保留预检日志；
不要直接降低 batch 或取消多尺度，否则会改变与原基线的对照条件。

正式训练沿用基线流程，不使用 -t 或 -r，不覆盖旧结果：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_mffe.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_seed0_run2" \
  --use-amp \
  --seed=0
```

注意：继承的基线配置使用服务器 SCB-S 路径。这次不另外生成 Colab
配置、不修改 notebook 流程、不修改数据集。若输出目录已有实验，使用
新 run 编号；本次不是断点续训。

## 训练后的同口径诊断

必须使用 MFFE 诊断配置和 **MFFE 自己的** best_stg2.pth。不要用 MFFE
架构读取原基线权重进行最终评估。下方 run2 对应显存修复后的重启；
原 run1 失败目录保留，不删除、不续训。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_mffe_diagnostics.yml \
  -r "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_seed0_run2/best_stg2.pth" \
  --test-only \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_seed0_run2/diagnostics_eval" \
  --use-amp \
  --seed=0
```

然后使用原分析工具，生成类别×尺度、query、PR 等同口径报告：

```bash
conda run --no-capture-output -n wq \
python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_seed0_run2/diagnostics_eval"
```

评价优先级：write AP、write→read / read→write 数量、write Precision/Recall、
类别×尺度 AP；再看总 AP、其他类别和定位是否退化。当前 110 轮实验使用
新输出目录（建议名称带 `epochs110`），与新 110 轮基线比较，不能直接对比旧 72/132 轮结果。一次实验不足以证明
稳定提升，更不能仅凭 APm 上升就宣称找到了信息丢失的网络层。

## 本地验证范围

回归检查覆盖：模块尺寸、门控归一化、首步非零梯度、CPU bfloat16
前向/反向、配置互斥、基线随机初始化和关闭时一致性、插入顺序、
DN/aux/原损失、optimizer 更新、checkpoint 加载、重计算梯度一致性和 deploy Encoder。
这不等于已完成 CUDA float16、4090 batch=32 或完整 132 轮训练。
