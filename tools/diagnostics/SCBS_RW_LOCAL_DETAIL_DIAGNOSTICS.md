# RW Local Detail：导出已有实验，不重新训练

当前实验整体AP为54.92，低于110轮基线三次均值55.16；write AP为48.35，
也低于基线均值48.50。不能因为高于基线Run2的47.61就判断模块有效。
这次只导出它自己的最佳权重，检查下降的构成，不修改模型、损失或分数。

## 同步与运行

在Mac终端同步诊断工具（不上传数据集/权重，不删除服务器文件）：

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/rw-detail-diagnostics-$(date +%Y%m%d-%H%M%S)" \
  tools/diagnostics/analyze_detection_diagnostics.py \
  tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh \
  tools/diagnostics/SCBS_RW_LOCAL_DETAIL_DIAGNOSTICS.md \
  configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

前提：服务器已有运行这次实验时的RW Local Detail源码、对应110轮训练配置、
当前完整query/混淆导出工具、SCB-S验证集及最佳权重。不要改用AQS或基线权重。
如果服务器文件另有修改，先核对；备份只保存被覆盖的同名文件，不自动合并。

在服务器运行一个命令：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh
```

默认读取：

```text
/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1/best_stg2.pth
```

输出在同一实验目录下面的`diagnostics_eval/`。
已有同名诊断目录就停止，不覆盖；可传入一个新的绝对路径：

```bash
bash tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh \
  '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1/diagnostics_eval_v2'
```

如果训练输出路径不同，用它实际的权重路径覆盖默认值，同时明确一个新输出目录：

```bash
RW_DETAIL_DIAG_CHECKPOINT='/实际实验目录/best_stg2.pth' \
bash tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh '/新的诊断输出目录'
```

脚本先检查原独立模块配置、110/100/batch32、验证集类别与图片、model/EMA严格加载；
用EMA执行`--test-only`，不启动训练。不固定占用7781端口。
实时日志同时写入`logs/evaluation.log`和`logs/analysis.log`。
任一步失败会停止、保留部分输出，不宣称导出成功。

脚本记录实际checkpoint epoch、权重/标注SHA256，以及局部分支最后投影的权重范数。
用户上传日志的整体AP最优epoch为102，导出后请核对实际权重与COCO结果。
严格加载只能验证结构匹配，不能辨别两个结构完全相同的不同run；路径与日志仍需核对。
权重范数只能证明参数是否非零，不能证明该分支有益或确定失败原因。

## 先看哪些结果

先打开`rw_local_detail_diagnostics_summary.txt`。汇总包括：

- 原COCO AP、各类AP，以及每类all/small/medium/large的AP、AP50、AP75、AR100。
- 原Validator的read→write、write→read、正确匹配、未匹配GT和未匹配预测数量。
- GT-best-IoU覆盖、低分/错类候选，以及是否存在另一高分同类候选。

主要文件：

| 内容 | 相对诊断目录的文件 |
| --- | --- |
| 类别×尺度COCO指标 | `analysis/coco_per_class_and_size.json` |
| 原混淆矩阵 | `evaluation/confusion_matrix/confusion_matrix.json` |
| 每个GT的query证据 | `analysis/gt_records.jsonl` |
| 冗余候选/候选缺口交叉统计 | `analysis/best_query_cross_classification.json` |
| 原COCO PR曲线 | `analysis/coco_pr_curves.json`及对应PNG |
| 分数分布、分数–IoU关系 | `analysis/score_distributions_and_quality.json`及对应PNG |
| 全部300个query的框与三类分数 | `evaluation/query_diagnostics/raw_queries/*.pt` |
| 完整性与权重标识 | `workflow_summary.json`、`export_inputs.json` |

COCO指标从同一次保存的`eval.pth`计算，maxDets=100，不使用0.5置信度筛选。
JSON指标为0~1，文本显示百分数；无有效GT的指标为null/--，不是0。
尺度依据原图COCO面积，边界按COCO包含规则；不是resize到640后的框尺寸。
GT计数是导出中的有效非crowd框数，若有ignore/crowd，与COCO有效统计口径需区分。

与110轮基线Run2诊断采用同一验证集、EMA、置信度0.5、IoU0.5比较。
重点看medium尺度的read/write AP/AR，双向混淆及真实检测缺口是否扩大；
“混淆减少但漏检增加”不能直接算改善。需要进一步定位具体图片时使用GT/query记录。
旧基线诊断也可用更新后的分析器读取已有raw queries和eval.pth，输出到新的分析目录，
无需重新训练或重新导出预测。

这些仍是描述性诊断：未匹配预测不等于背景，候选缺口不直接证明网络层的原因，
单次结果不能证明稳定性。请上传完整诊断目录，不只上传混淆矩阵图片。

## 本地回归测试

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_rw_local_detail_diagnostics.py
```

测试使用CPU生成权重/COCO小样本，并模拟启动流程；依赖同目录已有的
`test_scbs_110_run2_diagnostics.py`、`test_rw_local_detail.py`和`test_detection_diagnostics.py`测试夹具。
不会运行实际GPU验证或完整训练。真实服务器导出需另外执行上述命令。
