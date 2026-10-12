# SADF 内部诊断：固定最佳权重，不重新训练

目标：核对 SADF 的实际残差、尺度融合、双门控和对齐偏移，并对照 Baseline Run2。
本次只有新的诊断脚本，没有修改 SADF 的训练实现、损失、配置或权重。

## 先同步，再运行

Mac终端：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_SADF_runtime_audit.sh
```

服务器：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_SADF_runtime_audit.sh
```

默认诊断目录：

`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_SADF_seed0_run1/SADF_runtime_audit_run1`

旧目录或同名日志存在就停止，不覆盖、不续写。再次诊断请传入新的绝对输出目录，名称包含SADF。
本脚本没有替你同步或启动服务器进程，以上命令由用户执行。

## 运行内容

六次验证：原Baseline Run2 + 同一个SADF checkpoint的五个版本：

| 名称 | 训练后干预 |
|---|---|
| full | 完整SADF；额外捕获实际中间量 |
| no_alignment | P4/P5实际偏移置零，其余权重不变 |
| no_detail | detail置零，相关下游选择/门控自然重新计算 |
| equal_scales | P3/P4/P5融合权重均为1/3 |
| bypass | 返回SADF输入P3，绕过SADF输出修正 |

`bypass`不是原始基线：Backbone/Encoder/Decoder其他权重仍然来自SADF训练。
这些干预是固定权重的敏感性检查，会改变中间分布；不等同于从头训练的正式模块消融。
一分支关闭后AP变化不能单独证明其因果作用。

不创建优化器、不计算训练损失、不反传、不写checkpoint。模型eval、requires_grad=False，
所有前向在inference_mode中。临时forward接口和布尔开关用finally恢复，checkpoint settings buffer不改。
未改变普通训练路径；GT仅用于前向结束后的区域统计，不送入模型。

## 与原评估保持一致

- strict加载EMA，不存在EMA就报错，不回退到model。
- 使用配置中的Resize+ConvertPILImage、验证集collate、原sigmoid/Top300 query-class pair后处理。
- FP32推理匹配当前det_engine.evaluate，它即使收到--use-amp也没有启用推理autocast。
- batch-size默认8仅减少诊断显存占用；模型eval，不变训练批次。
- Run2权重和标注SHA256与用户之前提供的Run2诊断一致，否则停止。
- 逐GT校验annotation_id、category_id、原始框和area；不按最近框猜测GT身份。
- COCO AP使用所有后处理分数，正式COCO匹配。混淆矩阵使用原Validator：class-blind最高IoU贪心一对一，score/IoU均0.5。
- 候选覆盖只表示是否存在正确类别、score>=0.5、IoU>=0.5的选中pair，不是COCO TP或一对一Recall。
- 不把未匹配预测认作已经证实的背景或重复误检。

## 输出与含义

| 文件 | 内容 |
|---|---|
| inputs.json / complete.json | 配置、权重/标注指纹、来源、参数；只有complete=true才表示完整成功 |
| SADF_runtime_report.md | 六种版本AP、APm、read/write AP、中尺度write AP、读写混淆 |
| SADF_interventions.json | 各版本完整COCO/混淆/类别×尺度指标 |
| SADF_runtime_summary.json | 按类别×原始COCO尺度汇总全部GT实际中间量 |
| SADF_vs_Run2_paired_groups.json | 按候选覆盖gained/lost/both_covered/both_uncovered分组的中间量 |
| SADF_vs_Run2_gt_records.jsonl | 同一个GT的Run2/SADF预测与内部统计逐项记录 |
| SADF_runtime_examples.html / SADF_maps_*.png | 每组最多3个中尺度write样例；图中文字、框和热图均有对应GT ID |
| full/maps_*.pth | 仅中尺度write所在图像的降维标量图，可后续重新绘图；不是完整特征缓存 |
| 各版本/metrics.json、gt_records.jsonl、confusion_matrix/ | 各版本独立评估结果 |

按COCO标注原始area分组，不使用640resize后的area。32²/96²边界按COCO包含两侧。
内部统计只读与GT原框相交的P3格点范围（floor/ceil后裁剪），按每个GT等权汇总均值、中位数、p10、p90。
ignored GT仍写逐项文件，但不参与内部统计；crowd/退化框与原CocoDetection一样过滤。

残差相对强度：GT区域内 `RMS(P3_after-P3_before)/(RMS(P3_before)+1e-12)`。
不把alpha=0.1253解读为实际特征增加12.53%。

门控同时记录gate均值、门控前RMS、门控后RMS；不能仅根据gate大小比较不同分支贡献。
offset_x/y单位是P3像素；norm是二维长度。每个分量限制在±2，二维norm最大2√2。
near_bound_fraction表示分量abs(offset)>=0.95×上限的比例，只是一个描述性诊断统计。
偏移图中箭头是采样位移，不是经过验证的物体运动或身体注意力。

示例按每个候选覆盖变化组里相对残差较大的样本选取，存在选择偏差。
没有任何规定的权重/残差阈值能自动判定分支有效；需结合全部分组统计与正式AP。

## 只检查输入，不推理

服务器上可运行：

```bash
conda run --no-capture-output -n wq python tools/diagnostics/audit_sadf_runtime.py \
  --output-dir /tmp/SADF_preflight_unused --check-only
```

此命令校验数据/权重，不创建输出，不训练。

仅做小样本冒烟测试时加 `--max-images 8 --variants full` 并指定新输出目录。
子集结果会标注partial_validation，不可与完整验证集的AP比较。

CPU合成测试：`conda run --no-capture-output -n wq python tools/diagnostics/test_sadf_runtime_audit.py`。
合成测试是正确性验证，不是SADF有效性或服务器真实AP证据。
