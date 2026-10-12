# 110 轮基线 Run2 混淆矩阵

使用 `dfine_s_scbs_hrw_110.yml` 的原模型、数据与后处理，读取 Run2 自己的
`best_stg2.pth`。只运行验证，不训练、不重评分、不导出 query、不修改标注。
沿用 AQS-R1 的原 Validator：confidence >= 0.5、IoU >= 0.5，类别无关、最高
IoU 优先的一对一匹配。矩阵行是真实类别，列是预测类别。

## 同步新增文件（Mac 终端）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/run2_confusion_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110_run2_confusion.yml \
  tools/diagnostics/run_scbs_110_run2_confusion.sh \
  tools/diagnostics/test_scbs_110_run2_confusion.py \
  tools/diagnostics/SCBS_110_RUN2_CONFUSION.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器需已有近期使用的 `train.py`、原始 110 轮基线配置及现有混淆矩阵导出源码。
本次不更改这些文件，也不自动同步数据集或权重。

## 导出（服务器终端，先确认 GPU 空闲）

```bash
bash /home/a5/MyProject/wq_project/D-FINE/tools/diagnostics/run_scbs_110_run2_confusion.sh
```

默认权重：

```text
/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth
```

默认输出：上述 Run2 目录的 `confusion_eval/`。主要文件：

- `confusion_matrix/confusion_matrix.json`：原始计数、归一化矩阵、类别 Precision/Recall、权重来源。
- `confusion_matrix/confusion_matrix.txt`：可直接查看的矩阵。
- `confusion_matrix/confusion_matrix_counts.png`、`confusion_matrix_row_normalized.png`：两张矩阵图片。
- `validation_metrics.json`、`per_class_metrics.json`、`eval.pth`：同次验证的 COCO 结果。
- `logs/evaluation.log`：终端实时显示并保存的日志。
- `export_inputs.json`：评估前的权重/标注指纹；`workflow_summary.json`：仅在全部检查通过后标记完成。

旧输出目录存在时会停止，不覆盖、不删除。若需重新导出，传入新的绝对输出目录：

```bash
bash /home/a5/MyProject/wq_project/D-FINE/tools/diagnostics/run_scbs_110_run2_confusion.sh \
  "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/confusion_eval_v2"
```

`SCBS_RUN2_ENV`（默认 wq）、`SCBS_RUN2_GPU`（默认 0）、`SCBS_RUN2_CONDA` 可调整环境。
仅当原 Run2 权重移动了位置时，使用 `SCBS_RUN2_CHECKPOINT` 指定它的新绝对路径。
相同结构的 Run1/Run2/Run3 无法仅凭 state_dict 区分；必须确认所选文件确实是 Run2，
不要换成 Run1、132 轮、AQS 或其他模块的权重。

混淆计数不是 COCO AP。未匹配预测不一定是真背景，可能是重复、定位误差、未标注或
研究范围外目标。脚本不会自动排除教师或修正 GT。导出结束后上传整个 `confusion_eval/`
即可与 AQS-R1 按相同规则比较 read→write、write→read 和未匹配计数。
