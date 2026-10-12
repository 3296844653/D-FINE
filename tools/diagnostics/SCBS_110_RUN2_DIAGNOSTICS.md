# 110轮基线 Run2：完整预测诊断

目的：补全已经完成的Run2混淆矩阵所缺少的query级证据。
只读取既有权重验证一次，然后离线分析；不训练、不重评分、不修改GT、
不自动排除教师，不启用AQS或新局部细节模块，不运行训练集探针。

## 文件与范围

- 配置：`configs/dfine/dfine_s_scbs_hrw_110_run2_diagnostics.yml`。
  继承已有110/100完整诊断配置，模型/损失/数据/原后处理都不变。
- 脚本：`tools/diagnostics/run_scbs_110_run2_diagnostics.sh`。
- 原分析脚本仅修正报告标题：不再把任意实验都写成“Run1”，并显示实际权重与来源。
- 读取：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth`。
- 输出：上述Run2目录的`diagnostics_eval/`；不会覆盖`confusion_eval/`。
- 使用EMA、AMP、batch32、640验证输入，原后处理Top300 **query-class pairs**。
  同时保存每张图全部300个query的三类logits、原设备sigmoid分数和框。
  后处理Top300类别项不是300个不同query，不能把两者当成同一个统计。

运行前根据用户已上传的Run2完成记录核对指纹：

```text
checkpoint SHA256: 4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271
annotation SHA256: 623bb8478049dd69d32a92183b0a322bdb8f79f72e6f6f52e491d48923bd5d37
```

同结构的Run1/Run2权重无法仅靠模型结构区分，因此保留此严格检查；
不一致就停止，不自动接受其他权重/标注。路径移动可设置`SCBS_RUN2_CHECKPOINT`，
但内容必须还是这份Run2。model和EMA还会分别严格加载到原基线结构检查。
在验证前保存输入指纹，全部结束后复核；不完整结果不会显示成功。

## 同步（Mac终端）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/run2_full_diag_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110_run2_diagnostics.yml \
  tools/diagnostics/run_scbs_110_run2_diagnostics.sh \
  tools/diagnostics/analyze_detection_diagnostics.py \
  tools/diagnostics/test_scbs_110_run2_diagnostics.py \
  tools/diagnostics/SCBS_110_RUN2_DIAGNOSTICS.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器需要已有当前完整D-FINE源码、110轮基线配置、
`dfine_s_scbs_hrw_110_diagnostics.yml`以及现有query/混淆矩阵导出代码。
这些依赖正是此前Run1与AQS-R1完整诊断使用的版本；本次不修改它们。
以上同步不包含数据集、权重，也不覆盖训练配置/模型源码。

## 一键导出（服务器终端，先确认GPU空闲）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_110_run2_diagnostics.sh
```

使用`--test-only -r`，不是断点继续训练；不是`-t`。
standalone自动选端口，local-addr=127.0.0.1避开服务器主机名解析问题。
日志在终端实时显示，同时保存在`logs/evaluation.log`和`logs/analysis.log`。
有错误立即停止，保留部分结果，不输出“已成功”假提示。

若`diagnostics_eval/`已经存在，使用新目录，不删除旧文件：

```bash
bash tools/diagnostics/run_scbs_110_run2_diagnostics.sh \
  '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval_v2'
```

Conda环境默认`wq`、GPU默认0；可用`SCBS_RUN2_ENV`、`SCBS_RUN2_GPU`、
`SCBS_RUN2_CONDA`指定。只需要验证集图片与标注，不读取训练集。

## 先查看什么

1. `run2_diagnostics_summary.txt`：Run2总览，含权重路径、每类COCO指标、混淆、
   尺度分组和候选缺口。
2. `analysis/best_query_cross_classification.json`：区分类别正确低分/类别错误的
   best-IoU候选，以及是否另有已被原后处理保留的高分同类候选。
3. `analysis/gt_records.jsonl`：每个GT的图片名、annotation_id、类别、框、
   best-IoU query、三类分数、原后处理选择、候选缺口和一对一匹配信息。
4. `analysis/score_distributions_and_quality.json`、`coco_pr_curves.json`及相关PNG：
   分数分布、分数与IoU关系、官方COCO PR（从同次eval.pth读取）。
5. `analysis/prediction_records.jsonl`和`evaluation/query_diagnostics/nearby_predictions.csv`：
   未匹配/重叠预测的进一步线索，不自动认定为背景或真实重复误检。

类别×尺度×候选分组在`best_query_cross_classification.json`的`by_size_and_class`；
GT尺度数量在`gt_size_statistics.json`。原始张量保存在
`evaluation/query_diagnostics/raw_queries/<image_id>.pt`，后续可离线再分析。
只有校验完整后才生成`workflow_summary.json`且`complete=true`。
当前验证集应是1026图、5427有效GT、307800个query；计数由标注/导出核对，不写死。

## 如何解读

- A/C：best-IoU候选低分/错类，但另有高分同类候选。可能只是冗余候选问题。
- B/D：没有高分同类候选，值得进一步检查评分/类别混淆；不是直接的原因证明。
- G：所有query都没有达到IoU阈值，是空间候选不足的证据。
- 必须再结合一对一匹配：一个预测可能同时重叠两个GT，候选存在不等于检测成功。
- 未匹配预测还可能是重复、定位误差、漏标或范围外目标；抽样看图确认即可，
  不需要重新人工标注所有行。
- 最终分数包含LQE，未单独分解LQE贡献；一次最终query导出也不能定位到某个网络层。
- COCO PR与conf=0.5的自定义匹配计数规则不同，不能混用来计算同一种AP。

上传整个`diagnostics_eval/`即可继续判断；不要只上传混淆矩阵PNG。

## 本地检查

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_scbs_110_run2_diagnostics.py
```

测试只使用CPU合成权重/数据和模拟启动器，不进行真实服务器验证或训练。
