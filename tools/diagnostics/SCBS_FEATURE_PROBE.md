# SCB-S / run1：冻结特征与实际 Query 诊断

目的：检查同一个已训练的 D-FINE-S 中，哪些特征更容易区分 hand-raising、read、write，以及给**实际高分 Query**补充目标局部特征后，原本的分类错误能否减少。本实验不是新的检测模块，也不是 COCO AP 评估。

新增独立脚本 `tools/diagnostics/scbs_feature_probe.py`，保留旧 SCB-U 的 `stage_feature_probe.py` 和 `query_probe_export.py`，避免改变此前诊断的含义。新脚本复用原有工具中的配置读取、分类指标和 IoU 匹配函数。

## 运行内容

1. 严格加载 SCB-S/run1 的 `best_stg2.pth`，优先使用 EMA；冻结全部 D-FINE 参数，设为 eval。
2. 训练集、验证集分别前向一次，均使用无增强的 Resize 和 FP32。提取中间特征并缓存。
3. 只用训练集缓存训练 Linear 分类器，验证集仅评估；不根据验证集挑选训练轮次。每种对照固定训练 50 轮小分类器，使用 0、1、2 三个种子。这些不是 132 轮检测器训练。
4. 比较原分类、各阶段分类器以及 Query 加局部特征的分类器；保留原预测框、原分数和原候选配对。

HGNetv2 的官方 `use_lab: True` 保留；其他实验结构不允许启用。不会修改数据集、教师标注、原 checkpoint 或正式模型源码。

## 两组样本一定要分开看

| 分组 | Query 如何选择 | 用途 |
| --- | --- | --- |
| `selected` | 原 focal 后处理：所有 Query×类别的 top-300，再筛 score≥0.5；与 GT 按现有 Validator 进行类别无关、最高 IoU 优先的一对一匹配，IoU≥0.5 | **优先看这一组**，检验实际选中的高分预测是否获得分类改善 |
| `iou` | 不筛置信度，全部 Query 与 GT 做类别无关 Hungarian 配对：先最大化 IoU≥0.5 匹配数，再最大化 IoU 总和 | 检查候选可分性，含低分备用 Query；不是实际检测结果 |

`selected` 保留原后处理的 Query×类别候选。一个 Query 若产生两个类别预测，不擅自合并，也不把所选类别替换为该 Query 的 argmax。原分类、各个分类器的验证 GT 配对始终固定。

IoU 组中低分 Query 还分为三组：其 GT 已被 selected 正确匹配、已被 selected 错类匹配、没有 selected 匹配。改善第一组主要是改善备用候选，不能当成消除漏检。这里的“没有匹配”仅针对本次固定的匹配规则和阈值。

## 每组有九种特征对照

| 名称 | 输入 |
| --- | --- |
| `backbone_p2/p3/p4` | 对应 Backbone 特征图的 GT ROI 特征 |
| `encoder_p3/p4` | 对应 Encoder 输出的 GT ROI 特征 |
| `decoder_query` | 配对的最终 Decoder Query，256 维 |
| `query_only_capacity_control` | `[Q,Q,Q]`，768 维，只重复 Query、不增加图像信息 |
| `query_plus_pred_roi` | `[Q, ROI_P3(B_pred), ROI_P4(B_pred)]`，768 维；用原预测框采样局部特征 |
| `query_plus_gt_roi` | `[Q, ROI_P3(B_GT), ROI_P4(B_GT)]`，768 维；GT 几何位置上界对照 |

所有 ROI 使用相同的 3×3 ROIAlign、sampling_ratio=2、aligned=True 后平均池化。分类器均为三类 Linear；各阶段通道数及分类器参数量写入结果。归一化均值和标准差只从对应组的训练样本计算。

重复 Query 对照仅对齐输入维度、名义参数数量，**不能完全对齐有效秩、正则化和优化行为**。GT ROI 组使用真实框，是 oracle 对照，不可直接部署。预测框 ROI 组不需要推理时 GT，但本脚本的评估样本仍按 GT 选定，因此它本身不是部署验证。

## 4090 运行命令

配置为 `configs/diagnostics/scbs_feature_probe_run1.yml`，使用：

- 模型配置：`configs/dfine/dfine_s_scbs_hrw_132.yml`（原 132 轮基线快照，与旧 run1 权重对应）。
- 权重：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_seed0_run1/best_stg2.pth`。
- 数据：`/media/a5/5号机移动盘2/wq_datasets/SCB-S/`。

请从项目根目录运行，输出目录必须是新目录：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
python tools/diagnostics/scbs_feature_probe.py \
  --config configs/diagnostics/scbs_feature_probe_run1.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/feature_probe_run1_seed0"
```

无需 `torchrun`，无需 `-r` 或 `-t`。终端会每 25 张图输出提取进度，每 10 轮小分类器训练输出进度，输出主动 flush。使用服务器现有 `wq` 环境的 torch、torchvision、scipy、Pillow、PyYAML 和项目依赖，不下载新预训练模型。

也可分两步：先给上述命令加 `--mode extract`，成功后用相同配置、相同目录执行 `--mode train`。已存在的特征/分类器结果不会自动覆盖；提取若中断，请另选新目录重新提取。缓存校验会检查配置、checkpoint 和匹配记录，不能混用其他实验的缓存。

同步时至少需要新脚本、新配置、`configs/dfine/dfine_s_scbs_hrw_132.yml` 及其两个已有依赖 `tools/diagnostics/gt_box_classifier.py`、`tools/diagnostics/stage_feature_probe.py`。不要换成旧 SCB-U/run8 的配置。基线训练改为 110 轮不改变这里的旧 run1 诊断对象。

## 输出先看什么

- `summary.txt`：中文说明和汇总对照。包括原分类器、各探针的准确率、write 条件召回、读写混淆数、改对、改错、净改对。
- `summary.json`：各种子和样本组的完整指标，以及 `original_reference`；包含准确率、macro-F1、每类条件指标和混淆矩阵。
- `extraction_report.json`：原始高分预测的 4×4 匹配矩阵、匹配/低分覆盖数量、数据与权重标识。
- `probes/selected/<特征名称>/seed0/metrics.json`：实际高分组的详细统计；其中 `original_wrong`、`original_correct`、`read_write_confusion`、`medium_gt` 可直接看。
- 同目录的 `predictions.jsonl`：图片名、GT 编号、Query 编号、原预测类别、探针类别、原分数和 IoU。需要核查时再看，不要求逐条人工标注。
- `train_features.pth`、`val_features.pth`：冻结特征缓存；`head.pth` 是小分类器，不是 D-FINE 权重。

“改对”=原分类错、探针分类对；“改错”=原分类对、探针分类错。write 条件召回只统计**已匹配 GT**，不包括没有配对的 GT，不能与训练日志的检测 Recall 直接比较。Accuracy/F1 使用 0～1 的小数，例如 0.95 表示 95%；AP 不会在这次诊断中重新计算。

## 如何用来缩小下一步方向

1. 先看 `selected` 的 `decoder_query` 对比 `query_plus_pred_roi` 和 `query_only_capacity_control`：如果局部特征在三个种子上较一致地提高 write 条件召回、减少读写混淆且净改对为正，才支持继续研究分类专用的局部特征获取。
2. 如果只有 GT ROI 有效、预测框 ROI 无效，提示局部采样位置或几何对齐值得排查；不能直接断言 Backbone 丢失了细节。
3. 如果只在 `iou` 的低分备用 Query 上改善，而 `selected` 不改善，不能将其解释成消除了实际类别混淆。
4. 跨阶段可分性下降只是一条证据：ROI 位置、池化、通道数量、类别不平衡和小分类器优化都会影响结果，不能凭一次探针准确率锁定因果层。

提取采用 FP32、逐图 eval。与旧 batch=32/GPU 环境的导出相比，数值和 IoU 并列排序可能略有差异。解读前先检查 `extraction_report.json` 的原始高分矩阵是否与 run1 接近；若明显不符，先检查权重来源、数据、源码和预处理，而不是用错配结果指导结构修改。

本实验不判断未匹配预测是真背景还是教师/标注遗漏，不验证置信度排序，也不证明 AP 会提高。后续如获得积极证据，仍需独立检测实验验证。

## 本地检查

```bash
python tools/diagnostics/test_scbs_feature_probe.py
```

测试使用临时合成数据和小输入，不对真实数据集训练；检查实际 top-k 匹配、重复类别候选、空 GT、训练集归一化、局部 ROI、冻结模型严格加载以及输出防覆盖。
