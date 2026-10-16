# 四组独立的 AQS 基础结构候选

仅修改 `/Users/wq/Desktop/D-FINE`，同步到服务器后执行。四组不是一次叠加，也不是参数扫描：分别是 AQS-L2 R1 加一个完整结构。从头训练，110轮/停止增强100轮，batch32，seed0，原 HGNet 预训练、原优化器/增强/损失/EMA。不冻结 AQS，不用已有检测器权重微调。

这些是**可发展为论文创新点的研究候选**，并非已证明的创新或增益。需要检索近似结构、完整效果对照、组件消融、重复实验和计算开销，才能判断创新程度。第一轮只筛方向，不能写成稳定提升。

## 诊断依据

AQS 全模型审计表明：writing 684个GT均有IoU≥0.5候选，207个缺少同类高分覆盖；其中157个仍存在类别正确的候选，50个同时没有正确top1候选。验证集仍有write→read 239/684和read→write 156/2433。2570个高分未匹配预测不能全部当背景，其中存在重复、定位/标注范围等候选情形。类别、置信度和实例竞争需要分别检验。

Encoder线性探针比对应Backbone有更好的写类可分性，不支持直接宣称Encoder破坏特征。中尺度writing仅57个GT（large626），27个缺同类高分覆盖。不能只看APm、APs或只优化writing；应同时看hand/read/write、混淆两个方向、候选缺口、PR和总体AP。探针不是检测AP，oracle候选覆盖也不是TP。

## 四个完整结构

| 名称 | 位置 | 内部机制 | 与旧局部增强的关键区别 |
|---|---|---|---|
| ClassEvidenceTransport / CET | 初始化→L2 AQS之后→L3最终分类 | 原分类轴引导每类P3/P4证据，框外对照，带null拒绝的注意力，GRU类私有记忆，框稳定性控制运输；最终类别对比与共同分数残差 | 类私有证据与共享query并行保存，不再把一次ROI混合直接写进共享query |
| QueryContrastInitialization / QCI | Encoder Top300→Decoder L1 | 同一300个候选，框内/外语义对照，4个潜在区域的关系建模，零初始化query残差 | 非新TopK选择器、非单次ROI平均回填；检验早期候选内容而非增加数量 |
| GeometryAwareQueryCompetition / GQC | L3原分类+LQE之后 | 外观+几何关系，分别构建重叠/邻近实例关系库，排除自身并保留null，类别对比及共同score残差 | 显式跨query竞争，不采样ROI、不做硬NMS、不把所有未匹配预测标为背景 |
| PrototypeBoundaryRefinement / PBR | L3原分类+LQE之后 | 原分类轴+学习类向量引导自适应ROI取样，框内外证据对照，所有类别之间的差分/交互边界，竞争类别加权 | 不只是read/write反向加分；对三类分别收集、比较竞争证据 |

原分类权重归一化后作为“类锚”，不是用GT离线聚类出来的真实语义原型。潜在区域不是监督手部/物体检测。P3/P4来自当前Encoder记忆，未额外开启P2路由。质量描述为FDR分布熵/最大质量、类别模糊度等，不等于真实IoU或前景有效性。

三个评分结构CET/GQC/PBR在**固定权重开/关推理**中不改框；QCI修改初始query，可改变后续分类和框。CET在L2读到AQS后query，但不修改其共享路径。新分支只接普通query，不接GT生成的DN候选。最终输出teacher采用实际分类结果，遵循原D-FINE蒸馏/损失接口。

没有修改损失函数和权重，但训练后的匹配、原损失数值、特征和框仍可能间接改变，不能宣称只影响分类而训练定位完全不变。

新分支的输出末层均为零初始化，初始输出与AQS逐值一致；初始化不消耗原模型随机序列。缩放从0.1开始，通过tanh学习；这不是整个logit修正幅度的硬上限。类别对比残差去均值，另有可学习共同score项（QCI除外）。四个模块分别新增131423/101061/54341/103880参数。

思想借鉴而非整篇复现：[Highway Networks](https://arxiv.org/abs/1505.00387)的门控信息运输，[Relation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Relation_Networks_for_CVPR_2018_paper.html)的外观/几何关系，以及[GFLv2](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf)的框分布描述思想。未移植其损失。类私有运输、任务对照和组合的原创程度仍需进一步查新。

## 先同步（Mac）

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_BehaviorFour.sh
```

只同步清单文件，覆盖源码有远端备份，无`--delete`。不替换原基线配置、数据和旧结果。若本项目训练/诊断/队列仍运行，同步停止；不停止其他项目。如果原AQS R1配置或既有诊断依赖不存在，会报具体缺失文件。

## 再运行（服务器）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
tmux new-session -d -s scbs_behavior_four \
  'cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_BehaviorFour_110.sh --wait-gpu run1'
tmux attach -t scbs_behavior_four
```

无需等GPU手动启动：每30秒检查，计算进程退出、连续两次空闲后启动。不会自动杀任务；llama即使请求结束但服务器进程仍占卡，也继续等待。后台tmux不依赖Mac保持连接，但服务器重启/故障仍会中断。退出查看按Ctrl+B，松开后按D，不用Ctrl+Z。`duplicate session`时只attach，不新建同名会话。

四组顺序完整训练。实时日志＋`console.log`；训练前执行合成测试（服务器额外执行CUDA FP16），这些不是效果验证。每组独立输出：

`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_<模块名>_seed0_run1`

队列进度在`behavior_four_epochs110_seed0_run1/status.json`和`results.csv`，CSV指标单位0–1。旧输出或队列目录存在即拒绝，不覆盖、不自动resume，也不静默跳过。只跑一组可用：

```bash
bash tools/diagnostics/run_scbs_AQS_BehaviorFour_110.sh --wait-gpu run1 --modules ClassEvidenceTransport
```

先演练不会创建输出或连接GPU：

```bash
bash tools/diagnostics/run_scbs_AQS_BehaviorFour_110.sh --dry-run run1
```

## 自动导出与判断

训练完整后对`best_stg2.pth`导出AQS和新分支model/EMA参数、配置/源码/标注SHA256。由于D-FINE stage2重载可能导致checkpoint epoch字段滞后，最佳轮次以完整log.txt确定，并核对最佳EMA诊断AP。

同一最佳EMA分别开/关新分支，AQS始终开：导出混淆矩阵、全query记录、GT交叉分类、类别×尺度AP、PR/score分布，生成`<模块名>_on_off.json`和`<模块名>_diagnostics.zip`。压缩包排除权重和原始query二进制，保留JSONL、统计与图表。OFF是联合训练权重的推理消融，**不是单独训练的AQS基线**。

主要比较AQS R1，原110轮Run2做辅助。默认参考目录分别为`dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval`、`dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval`。可通过`SCBS_BEHAVIOR_AQS_REFERENCE` / `SCBS_BEHAVIOR_RUN2_REFERENCE`指定实际诊断根目录（含analysis和evaluation）。参考缺失/口径不兼容会明确记录`comparison_status.json`，不伪造比较、不丢弃新诊断。先用已有AQS诊断比较，无需重训基线。

若训练完成但尚无diagnostics_eval/off_eval/zip，可以使用`--export-only`导出；不重训、不覆盖已有诊断。部分诊断已存在时先保留并确认，再指定新的诊断目录人工导出，不能删除旧结果冒充新结果。

组件机制诊断通过`eval_behavior_refinement.py --mode no_memory/no_contrast/no_geometry/no_quality`；必须`--test-only -r`并写实际metadata开关。CET支持memory/contrast/quality；QCI支持contrast；GQC支持geometry/contrast/quality；PBR支持contrast/quality。这些仍是推理开关；论文严谨的组件消融应在相同训练设置下独立重训。

成功标准不是0.5阈值混淆计数单独减少。优先检查：AQS基础上AP/三类AP；双向读写混淆与同类高分候选缺口；hand是否退化；APm及各类GT数；PR和重复/低重叠预测；同权重ON/OFF；开销与重复实验。结果负向应如实保留。
