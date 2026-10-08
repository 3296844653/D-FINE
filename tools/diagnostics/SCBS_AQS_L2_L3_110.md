# SCB-S / D-FINE-S：独立参数 L2+L3 AQS，110轮

## 本次改变

```text
Decoder 1 → Decoder 2 → AQS_L2 → Decoder 3 → AQS_L3 → 最终分类
                                  └────────────────→ 最终框回归
```

两套 AQS 的 LayerNorm、MLP、选择阈值和残差参数**全部独立**，不是只把一个模块调用两遍。均从 tau=0.50、rho=0.05、固定温度 T=0.10 初始化，MLP 最后一层零初始化。不使用单层实验已经学到的参数。

对 l=2,3 分别计算：

\[
s_i^{(l)}=\max_c\sigma(z_{ic}^{(l)}),\quad
\tau_l=\sigma(\theta_l),\quad
u_i^{(l)}=\sigma((s_i^{(l)}-\tau_l)/0.1),\quad
h_i^{(l)}=\mathbf{1}[u_i^{(l)}\geq0.5],
\]

\[
g_i^{(l)}=\operatorname{stopgrad}(h_i^{(l)}-u_i^{(l)})+u_i^{(l)},\quad
C_l=\frac{\sum_i g_i^{(l)}Q_i^{(l)}}{\max(1,\sum_i g_i^{(l)})},
\]

\[
\widetilde Q_i^{(l)}=Q_i^{(l)}+\tanh(\rho_l)g_i^{(l)}
\operatorname{MLP}_l([\operatorname{LN}_l(Q_i^{(l)});C_l]).
\]

- L2：原第二层分类 + LQE 产生门控分数，增强后的 query 用于第二层辅助分类，并传入第三层（包括原 detached 残差路径）。第二层本次 bbox/FDR 不重算，但第三层定位可以受影响。
- L3：对经过 L2 增强后形成的第三层 query 再做最终分类增强，不直接改第三层本次 bbox/FDR。最终分数为原第三层分类头作用于增强特征，再加原 LQE。
- 普通 query 的定位蒸馏 teacher 使用第三层 **AQS_L3 前**的分类分数。这些特征已经包含 L2 的影响。没有第三条 baseline 反事实推理，也没有新增损失。
- 两个位置各自分开处理 DN/普通 query 上下文，避免把 GT-derived DN 特征聚合进普通组。
- 仍是三层、300个普通query、三类。原损失和权重、110轮/100轮停增强、batch32、seed0、数据路径及训练流程不变。
- 旧 L2/L3 单层配置与 `query_cls_refiner.*` 权重键名保持不变。新双层使用 `aqs_refiners.layer2.*` 和 `aqs_refiners.layer3.*`；严格加载不会把单层权重误当成双层完整权重。
- `aqs_apply_layers: [2, 3]` 是1-based。不要同时设置非默认 `aqs_apply_layer`，也不要更改 `eval_idx` 来定位AQS。

用户提供的单次 L2/L3 正向结果是本次试验的动机，不证明收益会叠加。阈值更低不等于实际参与上下文的 query 一定更多（两层分数分布不同），残差系数更大也不等于实际特征改动必然更强（还取决于各自 MLP 输出）。仍需检测结果/选中比例/特征改动证据。

## 参数量

原 L2 比 L3 的日志/deploy统计多2180个原有参数：第二层分类头771 + 第二层 LQE 1409。这是为了推理时计算门控而不裁掉第二层头，不是多加了另一个结构。

本次相比 L2 单层，真正再新增一套197,634参数的AQS。当前三类 S 的预期 deploy/日志参数量：**10,575,649（约10.58M）**。训练模型未裁剪的计数和deploy计数需使用同一口径比较。

## Mac同步

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_l2_l3_$(date +%Y%m%d_%H%M%S)" \
  src/zoo/dfine/dfine_decoder.py \
  src/solver/aqs_parameters.py \
  src/solver/det_solver.py \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_l2_l3.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_l2_l3_diagnostics.yml \
  tools/diagnostics/export_aqs_parameters.py \
  tools/diagnostics/test_aqs_l2_l3.py \
  tools/diagnostics/test_aqs_layer2.py \
  tools/diagnostics/test_aqs_refine.py \
  tools/diagnostics/SCBS_AQS_L2_L3_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器需要已有当前项目其他源码、单层AQS配置及数据。不使用 `--delete`，不删除数据/旧结果、不自动启动训练。

## 服务器小样本检查

确认 GPU 空闲，再检查真实CUDA FP16；不读取数据、不下载预训练权重：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
python tools/diagnostics/test_aqs_l2_l3.py
```

本地只有CPU，CUDA检查会明确skip；CPU通过不是4090显存或AP保证。

## 正式训练

新目录必须不存在。从头训练，不加 `-r/-t`，不加载 L2/L3 单层已完成权重：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_l2_l3.yml \
--output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_l2_l3_seed0_run1" \
--use-amp --seed=0
```

## 分层参数导出

训练每轮 `log.txt`、TensorBoard，以及 `aqs_parameters_final.json` / `aqs_parameters_best_stg2.json` 自动分别记录 L2、L3 的 model 与 EMA 参数。最后轮评估参数与最佳权重参数分开，不读可能停在第一阶段的 `last.pth` 代替最后轮。

也可以对已保存的权重单独执行（无需GPU、数据或推理）：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/export_aqs_parameters.py \
--checkpoint "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_l2_l3_seed0_run1/best_stg2.pth" \
--temperature 0.10
```

终端会分别打印 `model ...layer2 [L2]`、`...layer3 [L3]`，以及两个EMA分支。旧的单层读取命令也仍有效。

## 训练后检测诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
AQS_DUAL_RUN="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_l2_l3_seed0_run1"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_l2_l3_diagnostics.yml \
-r "$AQS_DUAL_RUN/best_stg2.pth" --test-only \
--output-dir "$AQS_DUAL_RUN/diagnostics_eval" --use-amp --seed=0
```

对照同设备/同流程的110轮基线、L2和L3，重点看AP、APm、read/write AP、AR100及双向混淆和Precision/Recall。不把132轮均值作为直接对照，不只凭单次小幅涨点确认机制或创新有效。
