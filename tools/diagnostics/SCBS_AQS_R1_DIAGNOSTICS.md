# AQS-R1：导出110轮最佳权重的诊断，不训练

只导出 R1：Decoder L2，初值 threshold=0.50、residual_init=0.10，固定 T=0.10。
配置初值不会覆盖 checkpoint 中的学习结果。使用 R1 自己的 `best_stg2.pth`，
不使用 baseline、L3、双层或 C1 的权重。单层权重键本身不能识别 L2/L3，
因此必须使用原实验对应的配置和权重路径；不要根据相同的键名混用权重。

## Mac同步

服务器需已同步当前 R1 源码和通用诊断/分析工具。本次新增下列文件；
不修改模型源码、训练配置、数据或权重，不自动连接服务器。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_r1_diag_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010_diagnostics.yml \
  tools/diagnostics/run_scbs_aqs_r1_diagnostics.sh \
  tools/diagnostics/test_aqs_r1_diagnostics.py \
  tools/diagnostics/SCBS_AQS_R1_DIAGNOSTICS.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

## 服务器一条命令

确认 GPU0 空闲后运行：

```bash
bash /home/a5/MyProject/wq_project/D-FINE/tools/diagnostics/run_scbs_aqs_r1_diagnostics.sh
```

默认权重：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/best_stg2.pth`

默认输出：同一个 R1 训练目录下的 `diagnostics_eval/`。只执行 `--test-only`；
保留原后处理、conf=0.5、IoU=0.5，与110轮基线诊断的阈值一致。
日志显示在终端并保存到 `logs/`；任一步失败即停止，保留部分结果。
输出目录已存在时停止，不删除、不覆盖、不自动恢复。

若之前已生成过该目录，换一个新目录再导出：

```bash
bash /home/a5/MyProject/wq_project/D-FINE/tools/diagnostics/run_scbs_aqs_r1_diagnostics.sh \
  '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval_v2'
```

如果 R1 权重另存，可用 `AQS_DIAG_CHECKPOINT=/绝对路径/best_stg2.pth` 指定；
必须仍是相同 R1 实验。环境可选 `AQS_DIAG_ENV`（wq）、`AQS_DIAG_GPU`（0）及
`AQS_DIAG_CONDA`（Conda可执行文件），不改变模型配置。

## 导出内容

- `evaluation/confusion_matrix/`：混淆矩阵 JSON、文本和 PNG。
- `evaluation/query_diagnostics/`：全部300 query的三类logits/分数/框，
  GT、最大IoU、原后处理选择、阈值前后记录和邻近候选；含 raw_queries/。
- `evaluation/validation_metrics.json`、`per_class_metrics.json`、`eval.pth`：原评估指标。
- `analysis/summary.txt`、`summary.json`：总览，含 read→write / write→read 计数。
- `analysis/gt_records.jsonl`、`best_query_cross_classification.json`：GT候选冗余与真实候选缺口证据。
- `analysis/gt_size_statistics.json`、`coco_pr_curves.json`、`score_distributions_and_quality.json`：
  类别×尺度、官方COCO PR曲线及分数/IoU关系。
- `aqs_parameters.json`：最佳权重中的model/EMA学习参数，不写回权重。
- `workflow_summary.json`：只在全部步骤和权重哈希校验成功后写入完成标记。

完成后把整个 `diagnostics_eval/` 下载到本地，与已有110轮run1诊断对比。
重点检查 R1 的 write AP 增益是否对应真实混淆减少、漏检减少或分数排序改善。
未匹配预测不是已确认背景，候选统计不是检测FP/FN；不自动改标注或排除目标。
