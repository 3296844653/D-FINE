# SCB-S：110轮Run1轻量诊断复核

使用已训练的 **110/100独立基线** 与自己的Run1 `best_stg2.pth`。
不重新训练，不改变模型、权重、置信度、类别、数据标注和旧132轮诊断。
Run1用于检查是否沿用旧问题方向；不能只用一次诊断证明三次运行的错误类型稳定。

## 服务器上一条命令

先同步本次文件及正常项目依赖，确认GPU0空闲，然后运行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_110_run1_diagnostics.sh
```

默认Conda环境为`wq`、GPU为`0`；可在命令前设置`SCBS_ENV`或`SCBS_GPU`。
脚本定位自身所在项目根目录，避免在configs目录中找不到train.py。
使用`torchrun --standalone`自动选择单机通信端口，避免固定7781冲突。
标准输出同时显示在终端和各步骤的`logs/*.log`中；任一步失败即停止，不报告假成功。
不会停止其他GPU进程，也不会自动等待它们结束。

固定读取：

```text
/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/best_stg2.pth
```

默认保存到：

```text
/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/diagnostics_recheck/
```

输出目录已存在时立即停止，不覆盖或续写。重新运行可显式传入新目录，例如：

```bash
bash tools/diagnostics/run_scbs_110_run1_diagnostics.sh \
  "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/diagnostics_recheck_v2"
```

## 执行内容及文件

1. `--test-only`评估：按110轮Run1的独立配置加载同一权重，导出COCO评估、各类AP、原Validator混淆矩阵及全部最终query。这里不是继续训练。
2. 离线分析：类别×尺度数量与AP、GT-best IoU、PR曲线、原分数分布与IoU关系，以及下面的best-IoU交叉分类。不替换分数、不计算自定义COCO AP。
3. 冻结推理：Encoder选前诊断估计、实际Top300初始参考框、最终Decoder框的覆盖率。使用新配置`configs/diagnostics/scbs_encoder_query_coverage_110_run1.yml`；旧`scbs_encoder_query_coverage_run1.yml`仍指向132轮Run1。

两次模型推理只遍历验证集；其余为CPU统计。Encoder-all额外运行冻结回归头，是**反事实几何估计**；覆盖率不等于检测Recall/AP，也不是故障层的因果证明。
详细定义沿用`SCBS_ENCODER_QUERY_COVERAGE.md`。

先看一个文件：`110_run1_recheck_summary.txt`，汇总分类、分数、尺度和Top300覆盖信息。
只有所有步骤完成且checkpoint SHA256、EMA/model来源、GT数量一致，才生成`workflow_summary.json`的`complete=true`。

关键结果：

- `evaluation/confusion_matrix/confusion_matrix.json`：原class-blind最高IoU一对一Validator结果，GT为行，预测为列。
- `analysis/summary.json`、`summary.txt`：保存的COCO各类AP、混淆、候选分组。
- `analysis/best_query_cross_classification.json`：A–G数量、各类别、各尺度×类别、无高分同类候选的原因分组。
- `analysis/best_query_cross_classification.jsonl`、`gt_records.jsonl`：每个GT的图片、annotation编号、框、类别、best-query、三类分数、其他正确候选存在与否，方便按需看少量图片。
- `analysis/coco_pr_iou0.5.png`、`coco_pr_iou0.75.png`、`diagnostic_tp_fp_scores.png`、`score_quality_relationship.png`：原PR/分数关系。
- `encoder_query_coverage/summary.txt`、`summary.json`：三个阶段按hand/read/write的覆盖率及跨阶段丢失/恢复数。

## 交叉分类在问什么

“高分同类候选”必须同时满足：与这个GT的IoU≥0.5、**GT类别项**原分数≥0.5、且该query×类别项确实被原后处理Top-k选中。
D-FINE后处理按query×类别选取，不是只取每个query的第一类别：即使GT类别排名第二，只要上述条件成立，也不能自动当作没有正确预测。

| 分组 | best-IoU query状态 | 高分同类候选是否存在 |
| --- | --- | --- |
| A | IoU足够，类别正确但低分 | 有；不能仅凭这个低分query判为漏检 |
| B | IoU足够，类别正确但低分 | 无；进一步区分全都低分或Top-k淘汰 |
| C | IoU足够，第一类别错误 | 有；包括同一query选中的高分GT类别项 |
| D | IoU足够，第一类别错误 | 无；有分类候选不足的证据 |
| E | IoU足够，第一类别正确且高分 | best-query的GT类别项未被Top-k选中 |
| F | IoU足够，第一类别正确且高分 | best-query的GT类别项已被Top-k选中 |
| G | 全部query均没有IoU≥0.5 | 无；几何覆盖不足 |

E仍需看是否有别的正确候选，不能直接当作漏检。A/C也不能自动当作重复误检，一个框可以同时重叠多个GT。
另存**类别一致、score-first、IoU=0.5的一对一高分匹配**，指出“存在正确候选但分配后没有匹配”的情况。
这与原Validator的class-blind最高IoU匹配不同，也不是COCO的ignore/crowd/maxDets匹配；不要把三个口径的计数混为一谈。

## 如何决定是否沿用132轮的改进方向

先比较110轮的read↔write混淆、write类中尺度AP/数量，以及B/D/G与无高分同类候选的规模。
再看Encoder Top300覆盖缺口是否经Decoder恢复。最终覆盖好但实际候选评分不足，与初始覆盖不足是不同现象。
旧132轮结果只是对照线索，不能替代本次权重的诊断；AMP批推理与FP32单图覆盖统计也可能有微小数值差异。
若主要错误方向相同，可以沿用；若比例明显改变，再调整模块目标。后续110轮模块只与110轮基线比较。

## 同步（Mac终端，不上传数据或权重）

以下命令保留服务器原文件的备份，不删除文件。假设服务器已有当前正常训练源码和配置依赖：

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir=".sync-backups/scbs_110_recheck_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml \
  configs/diagnostics/scbs_encoder_query_coverage_110_run1.yml \
  tools/diagnostics/run_scbs_110_run1_diagnostics.sh \
  tools/diagnostics/analyze_detection_diagnostics.py \
  tools/diagnostics/scbs_encoder_query_coverage.py \
  tools/diagnostics/scbs_feature_probe.py \
  tools/diagnostics/stage_feature_probe.py \
  tools/diagnostics/gt_box_classifier.py \
  tools/diagnostics/SCBS_110_RUN1_RECHECK.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

## 无数据测试

```bash
python -m unittest tools.diagnostics.test_scbs_110_recheck \
  tools.diagnostics.test_detection_diagnostics \
  tools.diagnostics.test_scbs_encoder_query_coverage \
  tools.diagnostics.test_scbs_baseline_profiles
```

只使用临时合成样例：验证110/132配置隔离、实际Top-k语义、A–G分组、阈值、类别×尺度、候选存在与一对一匹配的区别、输出防覆盖、权重来源检查及报告合并。
不读取真实训练数据、不下载预训练权重、不验证服务器上的实际模型精度。
