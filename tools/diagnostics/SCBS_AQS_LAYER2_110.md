# SCB-S / D-FINE-S：110轮，第二层 AQS 独立实验

## 位置与边界

原版本：`Decoder 1 → Decoder 2 → Decoder 3 → AQS分类增强 → 最终分类`。

本版本：`Decoder 1 → Decoder 2 → AQS → 增强query送入Decoder 3 → 最终检测`。

第二层原 query 先产生原 bbox/FDR 分布和分类 + LQE 分数，用这些分数进行 AQS 硬选择。AQS 的核心公式、DN 分组和参数初始化均不变：

\[
g_i = \operatorname{STE}\left[\max_c\sigma(z^{(2)}_{ic})\geq\sigma(\theta)\right],\quad
C=\frac{\sum_i g_iQ^{(2)}_i}{\max(1,\sum_i g_i)},
\]

\[
\widetilde Q^{(2)}_i=Q^{(2)}_i+\tanh(\rho)g_i
\operatorname{MLP}([\operatorname{LN}(Q^{(2)}_i);C]),\qquad
Q^{(3)}=D_3(\widetilde Q^{(2)},F_{encoder},B^{(2)}).
\]

训练时第二层辅助分类也使用增强后的 query；第二层 bbox/FDR 仍是增强前的结果。增强后的 query 同时进入第三层的普通路径及已有的 detached 残差路径，**最终分类和第三层回归都可能改变**。这不是只改一个训练时使用的辅助分类头，也不是截断第三层。

- `aqs_apply_layer: 2` 是 **1-based**，内部下标是 1；默认 `-1` 保留原最后层版本。
- 仍是3层、300个普通query、三类。`eval_idx` 仍为原来的 `-1`（内部下标2）。
- 最后一层不再执行第二次 AQS。每张图的普通 query 只经过一次第二层 AQS；训练时 DN/普通 query 分别聚合，没有 GT-derived DN 上下文混入普通组。
- 不改损失函数/权重，不叠加其他模块。第三层原分数作为普通 query 定位蒸馏 teacher；该分数包含第二层传入的增强特征，但第三层没有另一个分类修正。
- 使用同样的初始值：阈值0.50、固定温度0.10、原始残差0.05。**不把之前学到的0.53/0.14拿来初始化**，保证位置对照。
- 训练参数集合与最后层 AQS 相同。部署/统计时必须额外保留原第二层分类头和 LQE 用作门控，不能裁掉；相比最后层版的 deploy 模型保留2180个原有参数，并多算一次第二层分类/LQE。
- 原最后层版本的 state_dict 键不变，严格加载兼容。**权重本身无法区分插入位置，所以必须使用本实验自己的配置和权重。** 不把已训练的最后层 AQS 权重加载到第二层版本继续训练。

运行时参数 JSON/每轮日志还会记录 `decoder_layer: 2`。离线脚本只读旧权重时不能从 state_dict 恢复这个位置，须结合原配置/实验目录核对。

## Mac同步（不删除数据、不启动训练）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_layer2_$(date +%Y%m%d_%H%M%S)" \
  src/zoo/dfine/dfine_decoder.py \
  src/solver/aqs_parameters.py \
  src/solver/det_solver.py \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_diagnostics.yml \
  tools/diagnostics/test_aqs_refine.py \
  tools/diagnostics/test_aqs_layer2.py \
  tools/diagnostics/SCBS_AQS_LAYER2_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

要求服务器已具备当前项目其余源码与已有最后层 AQS 配置；不需要更新数据集。

## 服务器小样本检查

确认 GPU 没有其他训练，执行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
python tools/diagnostics/test_aqs_layer2.py
```

CUDA可用时会额外检查真实FP16及混合/空GT。Mac CPU检查不能替代4090显存或AP验证。

## 正式训练

新输出目录必须不存在。110轮、100轮停增强、batch32、seed0与原110轮对照一致，从头训练，不加 `-r/-t`。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2.yml \
--output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_seed0_run1" \
--use-amp --seed=0
```

## 训练后诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
AQS_LAYER2_RUN="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_seed0_run1"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_diagnostics.yml \
-r "$AQS_LAYER2_RUN/best_stg2.pth" --test-only \
--output-dir "$AQS_LAYER2_RUN/diagnostics_eval" --use-amp --seed=0
```

成功后可用原 `analyze_detection_diagnostics.py` 分析。对照110轮基线与110轮最后层 AQS，检查总AP、read/write AP、双向混淆、Precision/Recall。目的在于检验将 AQS 放在第二层并传给后续解码是否更有收益；没有预先保证精度提升，也不能仅凭一次结果认定瓶颈位置。
