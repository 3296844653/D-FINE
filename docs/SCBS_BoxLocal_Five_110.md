# BoxLocal-L2 基线上的五组独立结构实验

保持默认 BoxLocal-L2（dim64 / grid3 / offset0.15 / residual0.10）不变。
五组**分别**添加一个新模块，不是五个一起堆叠；AQS 全部关闭；不再给 L2 添加新分支。
不改损失、标注、三类顺序、110/100 轮和 batch32。现有配置、数据、权重和输出不覆盖。

## 为什么分散位置

BoxLocal-L2 单独训练的 mAP50:95 为55.6658%，write AP为49.1097%。现有两份组合日志显示，AQS+BoxLocal 串行run2为55.2758%/48.4098%，并行为54.7993%/47.5037%（整体AP/write AP）。因此本轮不继续叠AQS，也不只围绕L2改参数。单次实验不能确认精确原因，五个位置是待检验假设。

| 标签 | 新模块 | 实际插入位置 | 想检验的问题 |
|---|---|---|---|
| DDB | DirectionalDetailBackbone，方向细节骨干适配器 | 骨干P3输出、编码器之前，参考骨干P4 | 方向/局部高频细节是否在前端表达不足 |
| SCR | SemanticChannelRelation，语义通道关系 | 编码器P5经过AIFI之后、FPN/PAN之前 | 高层语义通道依赖能否改善后续多尺度融合 |
| CNF | CrossScaleNeighborhoodFusion，跨尺度邻域融合 | 原始FPN/PAN完成之后，P3读取P4邻域 | 细节尺度是否缺乏空间对应的上下文 |
| PRQI | PartRelationQueryInit，局部关系查询初始化 | Top300选出之后、L1之前、DN拼接之前 | 查询在进入解码前是否缺少框内局部关系 |
| RQC | ReliableQueryCompetition，可靠性查询竞争 | L3的原始分类与LQE之后，仅最终普通查询分数 | 重叠/邻近查询的信息与排序是否未充分利用 |

## 实现边界

- DDB是骨干**输出侧**适配器，不改HGNet内部stage；原骨干P4/P5输出保持原路径。
- SCR在P5做四头通道关系，不是新增空间全局注意力。输出继续走原FPN/PAN。
- CNF为每个最终P3位置读取对应P4的3×3邻域；带padding掩码与零值空候选。直接P4/P5输出不变，但改后的P3会进入整个解码器。
- PRQI用初始预测框在P3取5×5采样，四个潜在局部区域聚合，再融合区域差异/乘积/相对位置。没有手部/书本标注，不保证潜在区域有特定语义。框坐标detach；普通查询数和原TopK提案不变；不处理DN。
- RQC使用重叠/邻近双关系库与空候选。类别差异校正只看重叠关系；邻近关系只参与全类共享分数校正。可靠性由分类置信度和FDR分布集中程度近似，**不等于真实IoU或正确性**。当前框/查询/L1和L2分类不直接改变；训练中仍通过原损失反传。DN分数不变。
- 新增宽度默认32，残差尺度默认0.1，查询分块32。RQC两个可学习tanh尺度分别以0.05/0.10初始化。所有新增输出投影零初始化，构造时与原BoxLocal-L2等价；新分支构造不消耗原始模型的随机数序列。
- 原始BoxLocal-L2仍只在L2调用一次，dim64/grid3/offset0.15/residual0.10不变。
- 新增参数沿用优化器默认学习率0.00025；DDB输出适配器不放进预训练骨干的低学习率分组。原优化器配置不变。
- 不做额外正则或辅助损失，不按GT裁图，不按真实类别选查询。

这些是新设计的候选结构，**没有验证增益，也不保证论文创新性**。背景思想可参考[Deformable DETR](https://arxiv.org/abs/2010.04159)、[Swin](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)、[Relation Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Relation_Networks_for_CVPR_2018_paper.html)；不是复现上述方法。

## 使用：同步一次，运行一条命令

Mac桌面项目同步白名单（不连接服务器演练可加`--dry-run`）：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_boxlocal_five.sh
```

这次替换三个共享模型源码，需要当前D-FINE任务结束再同步；可加`--wait`自动等待。不会结束任何服务器任务。替换文件有独立备份，不同步数据/权重/旧输出。
五组配置依赖的`dfine_s_scbs_hrw_110_night_n4.yml`也在白名单内：服务器缺少时自动补齐；已有同名文件时核对SHA256并保留原文件，内容不一致则停止，避免覆盖已有实验配置。原始`dfine_s_scbs_hrw_110.yml`不被同步覆盖。其余已有依赖若缺失，会一次列出全部缺失项。
同步必须在Mac桌面项目执行，不是在服务器终端执行。传输后会检查服务器白名单文件及启动脚本语法，并显示实际服务器名、项目路径和“启动脚本已核对”。看到最后的“五组扩展同步完成”后，再到服务器运行训练命令；若仍提示找不到脚本，应先核对同步完整输出和服务器是否一致。

服务器的tmux终端中：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_boxlocal_five_110.sh all run1 --wait-gpu
```

按DDB、SCR、CNF、PRQI、RQC顺序各训练110轮，每组从原HGNet预训练开始联合训练，不从已有BoxLocal最佳权重续训，保证与原基线训练协议一致。默认seed0；`run2`只是换输出编号，不是换seed。要重复不同随机种子，用`--seed 1`等。

只跑一组：`bash tools/diagnostics/run_scbs_boxlocal_five_110.sh DDB run1 --wait-gpu`。
演练五组：`bash tools/diagnostics/run_scbs_boxlocal_five_110.sh all run1 --dry-run`。
默认Conda环境wq；可用`SCBS_BOXL_ENV`覆盖。
输出默认在`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw`，可用`--root /绝对路径`覆盖。

## 默认少做步骤

运行器不先跑单元测试、不做ON/OFF、不等待两次空闲确认。启动前只检查配置/数据/GPU/输出冲突。每组训练结束自动评估best_stg2的EMA（缺少时明确记录best_stg1回退），导出指标与混淆矩阵；不默认导出大体积query数据。需要完整query诊断时才加`--diagnostics`。

任一组失败即停止并保留日志；不会自动覆盖或续训。已有输出请选择新的run编号。`--wait-gpu`仅等待GPU计算进程，不会停止其他任务，也不构成多用户GPU预订锁。

## 结果留存与比较

每组输出：`dfine_s_scbs_hrw_epochs110_BoxLocalL2_<标签>_seed<seed>_<run>`。

- `console.log`、原训练`log.txt`、best_stg1/stg2.pth、last.pth：训练/恢复依据。
- `experiment_inputs.json`：解析配置、源码SHA256、数据标注SHA256、位置和seed；`experiment_config.yml`为入口配置备份。
- `diagnostics_eval/evaluation/validation_metrics.json`：同一最佳EMA权重的整体与分类AP。
- `diagnostics_eval/evaluation/confusion_matrix/`：混淆矩阵；P/R/F1使用conf0.5、IoU0.5微平均，不与COCO AR混用。
- `BoxLocalFive_epochs110_seed<seed>_<run>_<选择>/summary.md`与`results.json`：随完成实验逐组汇总，包含相对BoxLocal-L2的AP变化、write AP和read↔write误分。

先看整体AP、write AP、read AP、误分/漏检和复杂度是否协同改善，再对有正向信号的候选做多seed复验。不用单次最优值或同权重ON/OFF差异直接宣称稳定提升。
自动汇总中的基线是历史参考值，不是本次重新训练；正式比较前应核对数据、配置与源码记录。

本地640×640、batch1、部署结构的calflops统计（同历史基线口径，非实测延迟）：

| 模型 | 新增参数 | 部署参数(M) | FLOPs(G) |
|---|---:|---:|---:|
| 原BoxLocal-L2 | 0 | 10.2501 | 25.1146 |
| +DDB | 37,091 | 10.2872 | 25.4310 |
| +SCR | 21,892 | 10.2720 | 25.1321 |
| +CNF | 31,178 | 10.2813 | 25.4140 |
| +PRQI | 38,476 | 10.2886 | 25.2965 |
| +RQC | 17,934 | 10.2680 | 25.1853 |

本地实现检查（不由训练入口自动执行）：

```bash
python tools/diagnostics/test_boxlocal_five.py
```

本机没有CUDA，CUDA FP16检查需要服务器环境执行；本地CPU检查不是正式训练结果。
