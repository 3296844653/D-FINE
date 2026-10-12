# SCB-S：读写混淆 / 中尺度目标的5组独立结构实验

## 本轮边界

- 沿用 D-FINE-S、SCB-S 原标注、110轮、停止增强100、batch32、seed0。
- 5组分别训练，不是一次同时启用5个模块。不叠加AQS，不改VFL、匹配器、回归损失、数据增强、query数。
- 模块是待验证的研究方案，不保证提升，也不能仅凭模块名或单次AP宣称创新性。
- 单次表现和 Baseline110 Run2 比较；三次基线均值55.1565仅作总体精度参考。历史132轮/Colab结果不能当同设置基线。
- 原基线、已有权重、数据集和结果均不修改。新增参数由原检测损失训练。
- 只增强P3/局部ROI并不保证只影响中尺度GT；必须验证 medium read/write AP，不能拿APm或整体AP替代类别×尺度统计。

## 为什么是这5组

BoxLocal的query初始化/L2位置有正向响应，但不证明瓶颈仅在Decoder；EncoderCrossGate对APm有响应，但总体AP未改善。SADF整体收益很弱、中尺度write下降，不能据此直接认定“高分辨率信息越多越好”。据此分开检验细节保留、同尺度交互、框内外对比、多区域关系、读写双向证据。

| 组别 / 模块名 | 位置 | 真正改变的内容 | 新增参数 |
|---|---|---|---:|
| F1 DetailOnlyCrossGate | Encoder输出P3 | 原始Backbone P3的局部对比细节，经Encoder P3/P4门控后残差注入；P4仅控制门，不直接混入P4外观 | 101632 |
| F2 WindowInteractionP3 | Encoder输出P3 | 同尺度5×5窗口空间关系 + 局部对比，而不是继续混入低分辨率语义 | 54980 |
| F3 BoxScaleContrastL2 | Decoder第2层后 | 预测框内P3证据与框外P4环形上下文作对比，更新query再送L3 | 139106 |
| F4 PartInteractionL2 | Decoder第2层后 | 保留4个软区域描述及区域间外观/位置关系，而不是把ROI压成一个平均向量 | 150016 |
| F5 PairEvidenceHead | 最终分类，LQE之后 | 独立收集read/write两种局部证据，双向修正这两个logit | 63362 |

参数量按本地标准D-FINE-S配置实测；实际训练以日志为准。默认队列 F3→F4→F1→F2→F5。

### F1：DetailOnlyCrossGate — 语义门控，只补细节

设 raw Backbone P3 为 B3，最终Encoder输出为 F3、F4。投影统一到64维：

\[
D_k=DWConv_k(P(B_3)-AvgPool_k(P(B_3))),\quad k\in\{3,5\}
\]
\[
C=Up(P_4(F_4)),\quad (g_3,g_5)=\sigma(G([P_3(F_3),C,|P_3(F_3)-C|]))
\]
\[
F'_3=F_3+0.1W_o\phi([g_3D_3,g_5D_5,(g_3D_3)\odot(g_5D_5)])
\]

与旧CrossGate/SADF的区别：残差内容只来自raw P3；P4不能直接把整体人体语义作为外观加回P3。无P5上下文、无alignment分支。假设是避免上下文稀释细节，尚未验证。

### F2：WindowInteractionP3 — 同尺度局部关系

P3投影到64维，先做3×3深度卷积预处理，再在5×5窗口中做4头注意力：

\[
R_w=Softmax(Q_wK_w^T/\sqrt d+B_{relative}+M_{padding})V_w
\]
\[
F'_3=F_3+0.1W_o\phi([Merge(R_w)-P(F_3),P(F_3)-AvgPool_3(P(F_3))])
\]

不是完整Swin，没有shifted-window堆叠，不替换Encoder。3×3预处理跨相邻窗口边界，padding key被mask。用于测试同尺度局部空间关系是否有用，不能提前称其识别了手/笔/书。

### F3：BoxScaleContrastL2 — 框内证据与框外环境对比

使用L2已经预测的框 b2（坐标detach），框内P3的3×3点和框外P4的8个环形点分别query条件聚合。偏移有界，框外环的基础范围为框宽高的1.4倍。

\[
t=Pool(q_2,Sample(P_3,b_2,r_{in})),\quad c=Pool(q_2,Sample(P_4,b_2,r_{ring}))
\]
\[
g=\sigma(G([P(q_2),t,c,w_2,h_2])),\quad q'_2=q_2+0.1MLP([P(q_2),t,g\odot(t-c)])
\]

不是TopK选择器，不增加query数量。不同于BoxLocal单P3框内汇聚：显式对照框内外、不同尺度的内容。原L2 bbox在模块之前已计算，不被直接改写；L3的分类和定位都可能间接改变。

### F4：PartInteractionL2 — 保留区域与区域的关系

对L2预测框内5×5 P3样本，形成4个query条件软区域；四个象限中心仅作位置偏置，不是手/书/笔的人工监督。

\[
f_i=\sum_j a_{ij}v_j,\quad p_i=\sum_j a_{ij}r_j
\]
\[
R=Mean_{i,j}\,MLP([f_i-f_j,f_i\odot f_j,p_i-p_j])
\]
\[
q'_2=q_2+0.1MLP([P(q_2),Mean_i(f_i),R])
\]

与BoxLocal区别：保留多个区域及相互关系；没有用GT解剖标签、额外part loss。区域仍可能塌缩，四个区域不能未经可视化就命名为手、书、笔等。

### F5：PairEvidenceHead — 分别收集读/写证据，不固定提高write

最终预测框内5×5 P3样本，用query + 两个可学习证据向量分别聚合read/write描述 e_r、e_w：

\[
(\Delta_r,\Delta_w)=MLP([P(q),e_r,e_w,e_r-e_w,e_r\odot e_w])
\]
\[
g=\frac{s_r+s_w}{s_h+s_r+s_w}(0.25+0.75\cdot4\pi_r\pi_w),\quad
\pi=Softmax(z_r,z_w),\;s=Sigmoid(z)
\]
\[
z'=z+0.1g[0,\Delta_r,\Delta_w]
\]

g只从detach的原LQE-adjusted logits计算，不向原分数反传门控梯度；两项修正独立，不是强制相反的logit修正，也不是pairwise CE或固定write提分。假设是歧义样本需要分别验证两类证据。

固定权重前向时直接保留hand logit和bbox；训练仍可通过原VFL影响共享特征，不能承诺训练后hand/bbox完全不变。新增head只作用最终普通query，原aux/DN teacher不被这个最终修正改写。

## 共同实现约束

- 输出投影零初始化；新分支初始化不消耗原基线后续RNG，不改变原基线参数初始化。
- 残差缩放固定0.10，不是可学习alpha；这轮不同时扫参数。
- ROI坐标运算/grid_sample使用FP32，输出回到当前特征dtype。框坐标detach；无GT推理依赖。
- L2模块不直接处理DN query。但原attention mask允许DN关注普通query，所以下一层DN输出可能间接变化；不修改原mask强行隔离。
- GN/LN采用明确norm命名，进入原优化器的无weight-decay组；不改原优化器定义。
- 固定设置以整数编码保存为checkpoint buffer，避免EMA对浮点buffer的平均造成配置漂移。严格加载拒绝不同dim/scale/grid等设置，避免同名参数静默加载到错误配置。
- 默认finegrain_trial=none：原路径、原loss key和原checkpoint保持兼容。

## 如何判断实验是否对准问题

主要对照 Run2 的同一EMA best_stg2、原图COCO尺寸分组和相同阈值。

1. AP、read AP、write AP以及 medium read/write AP（优先）和APm；检查是不是write涨但read跌。
2. write→read、read→write两向混淆；不能只统计一个方向，也不能只以0.5阈值计数定论。
3. write正确高分候选覆盖与无覆盖缺口，以及GT交叉分类里的冗余/真正缺口。
4. unmatched预测、Precision/Recall/PR和定位质量；排除单纯提分造成更多误检的表象。
5. 有正信号后再重复/消融。单次+0.1AP不自动等于有效，先确认日志、配置和对应checkpoint。

自动导出混淆矩阵、原始query、GT-best-IoU、类别×尺度COCO指标和交叉分类；每组生成`diagnostics_eval/compare_BaselineRun2.json`。这些候选统计不是人工确认的错误原因，也不是因果层定位。

## 运行：先同步，再训练

Mac同步源是 `/Users/wq/Desktop/D-FINE`，不可从含未完成literature分支的工作目录同步。

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_finegrain_five.sh
```

服务器顺序运行5组：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_finegrain_five_110.sh
```

默认每组110轮，从原HGNet预训练开始完整训练，不是加载旧D-FINE best权重微调。训练日志实时显示并保存为console.log。先做小样本CUDA FP16反向检查；这不是完整batch32显存保证。

默认先把Run2重新验证一次（不训练）再做5组；Run2权重及验证标注SHA必须与已核验结果相同。若只想训练，可设 `FINEGRAIN_EXPORT=0`。缺少Run2权重时默认停止，不静默换基线。

只运行一组：

```bash
bash tools/diagnostics/run_scbs_finegrain_five_110.sh BoxScaleContrastL2
```

演练加 `--dry-run`，不训练、不创建目录。目录已存在则启动前停止，保留旧结果；可设 `FINEGRAIN_RUN_ID=2` 或只选择没运行的模块。队列遇到训练/诊断失败会停止并保留日志，不自动删结果、杀进程或断点重训。

结果目录统一为 `/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_<模块名>_seed0_run1`。

每组成本是一遍完整110轮；5组总时间约为单次耗时的5倍另加验证，不能保证一晚结束。

## 机制出处与范围

- 局部窗口关系参考 [Swin Transformer](https://arxiv.org/abs/2103.14030) 的窗口注意力思想；本实现不是完整Swin。
- 预测参考位置附近采样参考 [Deformable DETR](https://arxiv.org/abs/2010.04159) 的稀疏采样思想；不替换原D-FINE deformable attention。
- 多局部描述参考 [WS-DAN](https://arxiv.org/abs/1901.09891) 的判别区域表示思想；没有移植其crop/drop数据增强或原loss。

结构组合与本任务假设由本轮实验定义。已有注意力/采样思想不应包装成原创；“可作为论文创新点”还需文献区分、重复实验和机制证据。
