# SCB-S / run1：Encoder Top300 Query 覆盖率

独立诊断，不训练模型、不修改模型源码或旧探针，不改变分数、框、权重和数据标注。
默认分析旧 **132轮run1** 的 EMA `best_stg2.pth`，不是当前80/110轮的新基线。

## 三个统计位置

1. `encoder_all`：选前所有多尺度位置的诊断估计。原代码先按 `enc_score_head` 的最高类别logit选Top300，再仅对这300个位置运行 `enc_bbox_head`。本工具在正常推理结束后，用冻结的同一回归头额外计算全部位置的框，不改变选中的query或原推理。**这是反事实诊断，不是原程序已经输出的选前框。**
2. `encoder_top300`：实际传入 `TransformerDecoder` 的300个初始参考框，读取 `ref_points_unact.sigmoid()`。在 `pre_bbox_head` 和Decoder第一层修正之前，不能混用 `pre_outputs` 或最终 `pred_boxes`。
3. `decoder_final`：原模型最终300个 `pred_boxes`，不是后处理按query×类别选出的300条预测，不筛0.5置信度。

被动hook捕获特征、logits、实际Top300回归输入和Decoder参考框；不替换forward方法或返回值。
按原 `default` 选择规则重建grid编号，并与实际选中的特征逐项校验；不支持其他选择规则或不同query数量。
完整网格回归与Top300回归的FP32矩阵计算可能有微小差异，因此在全部位置结果中，被选中位置使用实际执行的Top300回归值，保证其为选前候选集合的精确子集。

## 统计定义

每个有效、非crowd GT独立寻找最大IoU：

\[
C_c^{(s)}(\tau)=\frac{1}{N_c}\sum_{g:y_g=c}
\mathbb{1}\left[\max_{q\in s}\operatorname{IoU}(B_q,B_g)\geq\tau\right].
\]

按 `hand-raising/read/write` 和全部GT分组，默认阈值0.3、0.5、0.75、0.9。
不要求预测类别正确、不做一对一匹配、不筛分数。**覆盖率不是检测Recall/AP，一个候选可覆盖多个GT。**
另外保存两种不同指标，不能与主覆盖率混用：

- `class_consistent`：只有argmax类别与GT相同的候选参与几何覆盖，仍不筛置信度，不是检测TP。
- `valid_anchor_only`：Encoder候选仅保留原有效anchor位置。主指标保留完整原候选池，包括可能无效的anchor；若无效位置带来覆盖，应查这项并核查逐GT记录。Decoder最终框不存在这里的anchor有效性过滤，其该项等于主几何覆盖。

阈值交叉统计：选前覆盖但Top300不覆盖（筛选损失证据）、Top300与最终均覆盖、Decoder恢复、Decoder丢失、Top300与最终均不覆盖。第一项与其余项不是互斥分组；其余四项互斥且覆盖全部GT。
选前框是冻结回归头的诊断估计，不能仅凭覆盖差异断言选择机制是因果瓶颈。
不自动排除教师，不修改类别、图片、标注或研究范围。

## 在服务器运行

先将桌面项目的新增文件同步到服务器。不能在基线训练占用GPU0时同时执行；也可 `--device cpu`，但会更慢。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
python tools/diagnostics/scbs_encoder_query_coverage.py \
  --config configs/diagnostics/scbs_encoder_query_coverage_run1.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/encoder_query_coverage_run1_seed0"
```

不需要 `torchrun`、端口、`-r/-t` 或新的训练轮次。只对验证集1026张图片推理，默认每25张输出进度。
输出目录非空时拒绝覆盖；中断后使用新目录。只有 `summary.json` 中 `complete: true` 确认完成。
如需训练集，把新诊断配置的 `splits` 改为 `[train, val]`，仍不训练模型。
如需80/110基线，复制诊断配置，同时改模型配置和对应checkpoint，不能把旧run1结果当成新基线结果。

## 输出文件

- `summary.txt`：每类、每个IoU阈值的三个位置覆盖率，以及筛选损失数、Decoder恢复数/丢失数。
- `summary.json`：完整计数、GT分母、类别一致覆盖、有效anchor覆盖、各阶段候选数量、配置和权重SHA256、权重来源及完整性标志。
- `val/gt_records.jsonl`：每个GT的图片名、annotation编号、三阶段最大IoU、最佳候选位置、原网格编号、框、类别与分数。
- `val/selected_queries.jsonl`：逐图实际Top300的网格编号、初始框和三类logits，以及对应最终query框/logits。框为归一化cxcywh。设 `save_selected_queries: false` 可关闭这项以减少文件大小。
- `resolved_config.json`：本次独立诊断配置。

不保存全部8400位置的逐框大文件，不生成AP或新的分类器权重，不修改旧诊断输出。
`checkpoint_epoch_metadata` 只是checkpoint中的 `last_epoch`；原训练可能重载stage1，不能据此推断训练总轮次/最佳日志轮次。

独立脚本依赖已有 `scbs_feature_probe.py`、`gt_box_classifier.py`、`stage_feature_probe.py` 及项目正常依赖。
需同步的新文件是脚本、新配置和本说明（测试可选）；旧依赖应与桌面项目一致。不要仅同步YAML而漏掉新脚本。

## 本地合成测试

```bash
python tools/diagnostics/test_scbs_encoder_query_coverage.py
```

验证实际初始参考框位置、候选集包含关系、输出与权重不变、EMA加载、类别分母、空GT、无效anchor、跨阶段交叉统计、异常hook清理及旧输出防覆盖。测试不会读取真实训练图片，也不下载预训练权重。
