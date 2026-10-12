# EvidenceContrastAQS：证据对比式 AQS 创新候选

这是基于项目 AQS 的新机制，不是已验证的创新成果，也不是某篇论文的官方复现。
当前先独立检验完整结构，不改检测损失，不与 SADF、BoxLocal 或旧版 AQS 并行叠加。
它通过 `use_aqs_refine` 的现有接口替换原 AQS 内部实现；旧配置仍使用 `legacy`，旧权重键不变。

## 1. 研究问题与预期

已有 AQS-L2 r=0.1 单次 AP=55.873，高于 110 轮三次基线均值 55.157；
write AP=49.742，write→read/read→write 的诊断计数也有减少。
这说明 query 内容交互值得研究，但不能证明 AQS 已定位了唯一瓶颈。

原 AQS 给所有接收 query 同一份筛选后平均上下文，并仅更新高分 query。
新假设是：**证据应按行为分开存储，并按接收 query 的歧义有选择地传递，而非只做全局平均。**
它不新增手部/手机的视觉信息；如果 query 根本没有这些信息，证据重组也可能无效。

```text
L2 Query + L2 分类分数 + L2 框/框分布（预测统计停止梯度）
             ↓
自适应源筛选 → 当前图像的软类别 × 槽位证据库
             ↓
每个 query：支持证据检索 + 竞争证据检索
             ↓
歧义条件接收门控 → 有界系数的残差更新 → L3 → 原检测输出
```

默认三类，每类 4 个槽位，共 12 个证据 token。300 个普通 query 不增不减。
图像之间不共享证据；训练 DN query 在独立分组内处理，绝不向普通检测 query 提供 GT 派生上下文。
不存在读取验证 GT、人工标签或类别原型的路径。

## 2. 对应代码的公式

输入 $q_i\in\mathbb R^H$、分类 logits $z_i$、cxcywh 框 $b_i$、四边的离散分布 logits。
这里所有 $z,b$ 及框分布统计作为条件均 detach；query 特征和模块参数仍由原检测损失学习。
这与旧 AQS 的分数门控可反传到分类 logits 有区别，必须在消融中核对。

### 源选择与预测可靠性代理

\[
p_{ic}=\sigma(z_{ic}),\quad s_i=\max_c p_{ic},\quad
\pi_{ic}=\frac{p_{ic}}{\max(\sum_c p_{ic},\epsilon)}
\]

\[
\tau=\sigma(\theta),\quad \tilde g_i=\sigma((s_i-\tau)/T),\quad
g_i=\operatorname{stopgrad}(\mathbb 1[s_i\ge\tau]-\tilde g_i)+\tilde g_i.
\]

前向为硬筛选，反向为 straight-through 梯度；它不是删除 query。
令 $m_i=\pi_i^{(1)}-\pi_i^{(2)}$、歧义 $u_i=1-m_i$。
四边分布的平均归一化熵为 $h_i$，集中程度 $\kappa_i=1-h_i$。

\[
r_i=s_i\frac{1+m_i}{2}\frac{1+\kappa_i}{2}.
\]

注意 $r_i$ 只是预测可信度代理，不是有效目标概率，也不等于真实 IoU。
尖锐的错误分类/错误框分布仍可能获得较大权重。

### 当前图像的软类别多槽证据库

\[
a_{ick}=\operatorname{softmax}_{k}\left(W_a[\operatorname{LN}(q_i)+f_b(b_i)]\right)_{ck},\quad
\omega_{ick}=g_i r_i\pi_{ic}a_{ick}.
\]

\[
M_{ck}=\frac{\sum_i\omega_{ick}\operatorname{LN}(q_i)}{\max(\sum_i\omega_{ick},\epsilon)},\quad
\bar b_{ck}=\frac{\sum_i\omega_{ick}b_i}{\max(\sum_i\omega_{ick},\epsilon)}.
\]

槽位分配是学习到的 soft assignment，不是聚类算法或手部/桌面部件标签。
每个 query 可以对多类证据做软贡献，不按预测 argmax 硬锁类别；也不保证槽位自动形成可解释部件。
证据库允许 query 自身参与，不宣称是严格 leave-one-out 的跨目标证据。

### 支持与竞争证据检索

每个 attention head 的基础关系分数为：

\[
e_{i,ck}=\frac{(W_q\operatorname{LN}(q_i))^\top W_k(M_{ck}+E_{ck})}{\sqrt{d_h}}
        +f_g(b_i,\bar b_{ck}).
\]

几何描述包含相对中心位移、对数宽高比例、与平均框的 IoU、归一化中心距离。
平均框不是真实对象位置，几何偏置不保证检索到同一个学生。
支持先验与竞争先验分别为：

\[
P^+_{ic}=\pi_{ic},\quad P^-_{ic}=\frac{1-\pi_{ic}}{C-1},\quad
A^\pm_i=\operatorname{softmax}_{ck}(e_{i,ck}+\log(P^\pm_{ic}+\epsilon)).
\]

\[
c_i^\pm=\sum_{ck}A^\pm_{i,ck}W_v M_{ck},\quad d_i=c_i^+-c_i^-.
\]

支持/竞争是**按当前预测分布的相对定义**，不是 GT 认定的正负证据。
差分是特征运算，不是新增 contrastive loss。
空证据槽位被 mask；整张图没有源候选时，检索结果为零且输出精确为 identity。

### 条件接收与残差更新

\[
v_i=g_i+(1-g_i)s_i u_i,\quad
\eta_i=v_i\,\sigma(f_r[s_i,m_i,\kappa_i,u_i,\|d_i\|/(1+\|c_i^+\|)]).
\]

实现中的范数使用稳定 RMS。低分但有歧义的 query 可以有限接收证据；极低分候选接收幅度很小。
没有任何证据时，全部接收门为零。

\[
q'_i=q_i+\tanh(\rho)\eta_i f_{\Delta}
   [\operatorname{LN}(q_i),c_i^+,d_i,c_i^+\odot d_i].
\]

最终投影零初始化，因此不移动 baseline 原参数随机初始化，初始输出与 baseline 完全一致。
有界的是残差系数和接收门，而不是 MLP 输出或整个特征增量的范数。
第一步只更新最终投影；随后阈值、槽位、检索及接收参数开始获得非零梯度。
原始初始化：$\tau_0=0.5,\rho_0=0.1,T=0.1$，实际初始缩放为 $\tanh(0.1)\approx0.09967$。

L2 本层已经计算出的 bbox 不直接改变；修改后的 query 会进入 L3，所以最终 bbox 可以间接受影响。
没有把它说成严格的“仅分类不影响框”的模块。

## 3. 规模与消融接口

新增参数 245,747；原始训练模型总参数 10,468,756（原始 baseline 10,223,009）。
原 AQS-L2 模块为 197,634 参数，因此新机制相对旧 AQS 模块多 48,113 参数，而不是额外叠加一份 AQS。
部署模型总参数 10,426,128；相对 baseline 部署的 10,178,201，多 247,927（模块245,747＋保留L2分类/LQE头2,180）。
raw/deploy 数量不同，比较参数量时统一口径。
新增银行构造/检索的规模为 $O(QCS)$ 的 token 交互；不新增 $Q\times Q$ 关系矩阵。
原 decoder 的 self-attention 仍存在，不能声称整个检测器是线性复杂度。
真实 FPS、峰值显存、训练时间需要在同一 4090 上测量，尚未验证 ONNX/TensorRT。

只写一份当前完整实验配置；后续通过这些默认开启的接口做结构消融：

| 接口 | 关闭后检验什么 |
|---|---|
| `aqs_evidence_slots: 1` | 多槽位是否比单类均值提供更多有效内容 |
| `aqs_evidence_geometry: false` | 去掉槽位分配的框嵌入及检索几何偏置 |
| `aqs_evidence_quality: false` | 源权重只保留分数，不使用 margin/框分布集中程度 |
| `aqs_evidence_contrast: false` | 只使用支持上下文，不使用支持−竞争差分及交互 |
| `aqs_evidence_adaptive_receiver: false` | 只让原始 AQS 高分候选接收更新 |

开关消融保留参数/计算图（除 slots 维度改变），因此不是省计算模式。
此外必须补充相同容量 MLP/普通 bank-attention 对照、预测统计 detach 的控制对照；
否则不能排除收益只来自参数量、接收范围变化或优化差异。

## 4. 相关工作与创新边界

不能把 attention、软证据库、几何关系、straight-through 或支持/竞争概念单独声称为原创。
当前检索并非穷尽式查重，以下只构成需要重点比较的近邻方法：

- [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575)：外观与几何的对象关系已有研究。
- [Adaptive Clustering Transformer](https://arxiv.org/abs/2011.09315)：压缩成 prototype 后进行 attention 的思想已有研究。
- [QR-DETR](https://openaccess.thecvf.com/content/ACCV2024/html/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.html)：query 路由/选择并非新概念。
- [Dual-R-DETR](https://arxiv.org/abs/2512.13876)：根据外观、置信度和几何设计竞争/合作的 query 关系；其方式是 self-attention 的 pairwise routing 偏置与训练双分支，而非本实现的类别/槽位库检索。
- [BS-O2G](https://arxiv.org/abs/2609.26092)：使用预测感知稀疏图做 query 协作及 backward sharing；本实现没有该图，也没有反向梯度路由，但同样研究 query 证据利用，必须认真比较。

拟验证的贡献是：**在固定数量的检测 query 中，将自适应源选择、预测条件的类别/槽位证据组织、
支持−竞争特征对比、歧义条件接收统一为一个中间 decoder 残差机制，用原检测监督改善视觉相近行为。**
机制差异不自动等于论文级创新；需要完整文献对比、可解释证据和稳定增益。

## 5. 训练、导出与判断标准

Mac 同步（只同步清单、备份被替换文件；不改服务器基线、数据、旧结果）：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_EvidenceContrastAQS.sh
```

服务器启动一组，从头训练，不续训旧 AQS：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_EvidenceContrastAQS_110.sh
```

输出：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_EvidenceContrastAQS_seed0_run1`。
110/100 轮、batch32、seed0 与 baseline 一致；实时终端与 `console.log` 双份输出。
启动器先跑合成测试（CUDA 可用时包含 FP16），再训练→导出 best_stg2 的 model/EMA 参数→完整诊断→离线分析。
合成测试不是精度证据；当前本地 CPU 可测，4090 测试由服务器启动器执行。

AP 同时对照 baseline 三次均值与旧 AQS-L2 R1；诊断按用户要求统一对照 Baseline Run2。
重点不是“分数跨过 0.5 的数量增加”，而是：

1. write AP 与整体 AP 能否超过旧 AQS，而不仅超过一次 baseline；read AP 是否保持。
2. write→read 和 read→write 是否同时减少；不能一边改对一边等量改错。
3. APm、中尺度 write AP 是否改善（该类 medium GT 只有57个，必须关注稳定性）。
4. 正确高分候选缺口是否减少，同时 Precision/PR 和未匹配预测是否恶化。
5. 若有正信号，再进行重复实验、拆分消融、相同容量控制、另一数据集验证及运行效率测量。

未匹配预测不自动等于背景，低分最佳 IoU query 不自动等于漏检；全部沿用原诊断的交叉覆盖口径。

最佳 checkpoint 参数导出文件：`best_stg2_EvidenceContrastAQS_parameters.json`。
每轮/最佳/结束的旧 AQS 标量日志接口仍自动记录 τ、ρ、tanh(ρ)、T 和作用层。
新增固定设置持久化到 checkpoint，专用导出器不需要手工猜温度或模块名称。
评估/恢复须使用训练时相同的配置；加载时检查固定机制设置及作用层，避免温度/开关与导出报告不一致。
运行时 `return_aux=True` 可检查源门、槽位分配、银行权重、支持/竞争 attention 与接收门；
这些量只能说明机制行为，不能证明看到了手或正确的书本关系。
