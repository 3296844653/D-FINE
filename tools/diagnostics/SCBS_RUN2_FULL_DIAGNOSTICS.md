# SCB-S 110 轮 Run2：后续诊断的统一基线

本入口只加载已经训练完成的 Run2 `best_stg2.pth`，使用 EMA 做一次验证和离线统计。
不重新训练、不调整预测分数、不改模型、损失或标注，也不覆盖旧混淆矩阵。

## 服务器运行

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_110_run2_diagnostics.sh
```

默认权重：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth`

默认新输出目录：同一 Run2 目录下的 `diagnostics_export1/`。
若这个目录已存在，脚本停止并保留旧结果。需要另导出时显式传入新目录：

```bash
bash tools/diagnostics/run_scbs_110_run2_diagnostics.sh \
  "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_export2"
```

权重或数据搬迁时可用 `SCBS_RUN2_CHECKPOINT` 指定新权重路径；脚本仍校验已经确认的
Run2 权重及验证标注的 SHA256，不会把另一实验作为 Run2。诊断配置继承正常
`dfine_s_scbs_hrw_110.yml`，只开启导出开关；该训练配置保持不变。

## 导出内容与上传

- `evaluation/confusion_matrix/`：原 Validator 混淆矩阵、图片。
- `evaluation/query_diagnostics/`：每图全部 300 个 query、原分数、真实 Top-k 选择、GT 覆盖。
- `analysis/gt_records.jsonl`：图片/GT/query 编号、best-IoU、类别和原分数。
- `analysis/best_query_cross_classification.json`：A–G 候选分组、各类别与各尺度的高分正确候选缺口。
- `analysis/coco_per_class_and_size.json`：每类 × small/medium/large/all 的 COCO AP/AP50/AP75/AR100。
- `analysis/coco_pr_curves.json`：从保存的 COCO 评估读取的 PR 曲线。
- `analysis/score_distributions_and_quality.json`：诊断分数分布与 IoU–score 关系。
- `workflow_summary.json`：Run2 身份、权重/标注指纹、完整性检查结果。
- `logs/`、`run2_diagnostics_summary.txt`：实时日志和易读汇总。

将整个 `diagnostics_export1` 文件夹上传，后续模块诊断统一与这份 Run2 比较。
不得用 Run1 替代缺失的 Run2 query 结果，也不能把候选缺口直接当成 COCO FN。
未匹配预测不等于人工确认的背景误检。单次模块结果不证明稳定性或某一层的因果瓶颈。
