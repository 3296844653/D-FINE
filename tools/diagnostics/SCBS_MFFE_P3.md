# SCB-S / D-FINE-S：MFFE P3-only 消融

这次只检验“移除 P4 的直接增强，是否比 P3+P4 MFFE 更合适”，不预先认定 P4 是下降的原因，也不保证 P3-only 会提升。模块参数不变，当前各组统一继承 110 轮基线。

## 唯一结构变化

原实验：投影 → AIFI/FPN → **P3、P4 分别 MFFE** → PAN → Decoder。

新实验：投影 → AIFI/FPN → **仅 P3 MFFE** → PAN → Decoder。

新增 `HybridEncoder.mffe_p3_only`，默认 `false` 保持原 P3+P4 的前向、参数名和初始化。设为 `true` 时只创建 `mffe[0]`，不创建闲置的 P4 分支、参数或优化器状态；保留 P3 分支原有初始化。关闭 `use_mffe` 仍是原基线。

P3 分支的计算不变：

\[
\widetilde T_3=T_3+\alpha_3\operatorname{GN}\left(W_o\sum_{k\in\{3,5,7\}}\pi_{3,k}\odot B_{3,k}\right),
\qquad \alpha_3=0.5\tanh(a_3).
\]

融合前的投影 P3 提供细节，融合后的 FPN P3 提供语义，引导 3×3、5×5、7×7 深度卷积分支的空间 softmax 权重。初始 alpha=0.1；`mid_channels=64`、GroupNorm、激活重计算机制、FPN 后/PAN 前的位置均保持不变。

原 FPN P4 不直接经过 MFFE，但 PAN 会接收增强、下采样后的 P3，因此最终 P4/P5 输出仍可能变化。共享 Encoder 特征也可能同时影响分类和回归；这不是纯分类分支或完全隔离 P3 的实验。

新增参数从原 P3+P4 的 112,008 降为 56,004。当前 deploy/profiler 参数口径为：基线 10,178,201，原 MFFE 10,290,209，P3-only 10,234,205。不是训练态参数口径，也不据此保证 CUDA 显存容量。

## 配置

训练：`configs/dfine/dfine_s_scbs_hrw_mffe_p3.yml`。

诊断：`configs/dfine/dfine_s_scbs_hrw_mffe_p3_diagnostics.yml`。

新训练配置直接继承 `dfine_s_scbs_hrw.yml`，只配置当前模块：

```yaml
HybridEncoder:
  use_mffe: true
  mffe_mid_channels: 64
  mffe_alpha_init: 0.1
  mffe_checkpoint: true
  mffe_p3_only: true
```

数据、类别顺序、输入/多尺度流程、batch=32、损失、300 queries 保持一致；当前各组统一为 110 epochs、epoch 100 停增强。不要同时调 alpha 或宽度，否则无法单独判断移除 P4 的影响。

## 同步到服务器

以下命令在 Mac 终端运行，不使用 `--delete`，不传数据集和已有权重。若服务器有其他未同步的源码修改，先保存这些修改，再同步共享源码。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw.yml \
  src/zoo/dfine/hybrid_encoder.py \
  src/zoo/dfine/medium_finegrained.py \
  configs/dfine/dfine_s_scbs_hrw_mffe_p3.yml \
  configs/dfine/dfine_s_scbs_hrw_mffe_p3_diagnostics.yml \
  tools/diagnostics/test_medium_finegrained_p3.py \
  tools/diagnostics/check_mffe_cuda_memory.py \
  tools/diagnostics/SCBS_MFFE_P3.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器可先做无数据、无下载的 CPU 检查：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq python tools/diagnostics/test_medium_finegrained_p3.py
```

沿用显存预检工具，只需指定新配置：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
python tools/diagnostics/check_mffe_cuda_memory.py \
  --config configs/dfine/dfine_s_scbs_hrw_mffe_p3.yml
```

预检只训练随机权重模型的少量步，检查实际数据 batch=32 和训练分辨率，丢弃模型，不保存实验权重。抽样通过不保证所有训练批次都能通过。

## 完整训练

保持与基线相同的训练流程，不用旧 MFFE 权重热启动，也不使用 `-t` 或 `-r`：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_mffe_p3.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_p3_seed0_run1" \
  --use-amp \
  --seed=0
```

使用新输出目录。若该目录已存在，请更换 run 编号，不要覆盖旧结果；有其他 torchrun 作业时也要使用不同端口。

训练日志配置应包含 `mffe_p3_only: True`。参数量按现有 profiler 口径约 10.23M，而非原 MFFE 的 10.29M。

## 训练后的同口径诊断

必须用本次 P3-only 架构和本次权重，不能把原 P3+P4 的 checkpoint 当作 P3-only 结果。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_mffe_p3_diagnostics.yml \
  -r "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_p3_seed0_run1/best_stg2.pth" \
  --test-only \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_p3_seed0_run1/diagnostics_eval" \
  --use-amp \
  --seed=0
```

随后沿用原诊断分析工具：

```bash
conda run --no-capture-output -n wq \
python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_mffe_p3_seed0_run1/diagnostics_eval"
```

比较三组时，基线、P3+P4 MFFE、P3-only 必须都是同一 110 轮流程，不能拿旧 72/132 轮结果直接对比。各组使用新输出目录，建议名称带 `epochs110`。先看 write AP 和双向 read/write 混淆，再看 write Precision/Recall、总 AP、其他类别和类别×尺度表现。

仅超过原 MFFE、仍落在基线波动范围内，只能说明负面影响减小，不能认定有效。若 P3-only 也未降低目标错误、未提高 write AP，暂停 MFFE 方向，而不是继续叠加模块。一次正向结果仍需复验。

## 验证边界

本地 CPU 合成测试覆盖：默认双分支兼容、P3 初始化/RNG、一分支参数、FPN/PAN 接入、关闭和零残差的基线一致性、CPU bfloat16 重计算梯度、完整 D-FINE-S 损失和优化器、严格 checkpoint 加载、诊断配置。

这些不是服务器 CUDA/AMP 完整训练或 AP 改善证据；4090 上的训练与效果仍需实际验证。
