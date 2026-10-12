# SADF：尺度对齐细节融合（Scale-Aligned Detail Fusion）

这是针对SCB-S中尺度细粒度行为的完整Encoder候选机制，不是已经证实的原创方法或有效创新点。
没有训练结果之前，不能承诺AP提升，更不能把若干已有操作的组合自动等同于充分创新。

## 研究问题与依据

Baseline Run2中尺度write AP=32.50，整体write AP=47.61；EncoderCrossGate的中尺度write AP=35.44，
但整体AP约55.14，未超过110轮基线三次均值55.16。它提供的是尺度相关的响应线索，
不是“Encoder是唯一瓶颈”的因果证据。旧MFFE、高分辨率残差等也没有给出稳定的整体改善。

SADF检验一个具体假设：**浅层对比细节、跨尺度语义对应关系与融合选择需要协同建模；
单纯上采样并门控细节，可能无法同时避免细节弱化和无关上下文注入。**
误对齐/无关上下文目前仍是假设，需要消融与可视化，不宣称已被这些日志证明。

## 位置与边界

```text
Backbone P3 -> 原input_proj -> R3（浅层细节来源）----------------------┐
                         原AIFI + FPN + PAN -> P3/P4/P5            │
                                                  │               │
                       SADF：对比细节 -> 对齐 -> 尺度竞争 -> 双门控融合
                                                  │
                                         P3' / 原P4 / 原P5
                                                  │
                                      原Top300 + 原Decoder + 原损失
```

- 在原始FPN/PAN全部完成后只直接修改P3，不向PAN反馈，当前forward的P4/P5保持原值。
- 不增加P2、query、类别头或损失，不叠加AQS/BoxLocal，不使用GT类别、框、COCO尺度分组。
- P3是stride8特征层，不等于COCO中尺度目标。所有大小/类别都可能受影响。
- 共享训练和Decoder都可以改变分类及最终bbox；不是“只改分类、定位不变”。

## 完整机制与公式

记原投影P3为R3，最终Encoder输出为P3/P4/P5；投影phi压至d=64。
F=phi3(P3)，S=phiR(R3)，C_l=phi_l(P_l)，l=4,5。各phi独立学习。

### 1. 有符号、多范围局部对比

\[
D_k = DWConv_k\left(S-AvgPool_k(S)\right),\qquad k\in\{3,5\}
\]
\[
(u_3,u_5)=Softmax\left(h_D([F,|D_3|,|D_5|])\right),\quad D=u_3D_3+u_5D_5
\]

保留正负局部变化，不将高频强度当作行为证据或手部定位标签。
平均池化在边界排除填充值，常量特征不会因零填充产生假边缘。

### 2. 对比细节条件的有界跨尺度对齐

\[
\Delta_l=b\tanh\left(h_l([F,Upsample(C_l),D])\right),\quad b=2\text{（P3像素）}
\]
\[
\widetilde C_l(x)=BilinearSample\left(C_l,\,g(x)+[2\Delta_l^x/W_3,2\Delta_l^y/H_3]\right)
\]

offset顺序是x/y，坐标以P3像素定义，固定上限不是coarse像素也不是整个图像的归一化距离。
原生PyTorch grid_sample，不依赖编译DCNv2扩展；align_corners=False、border padding、FP32采样。
有界不保证不会取到相邻学生，只限制位移幅度；是否更符合行为区域须观察采样位置。

### 3. 差异感知的尺度竞争

\[
(w_3,w_4,w_5)=Softmax\left(h_S([F,|F-\widetilde C_4|,|F-\widetilde C_5|,|D|])\right)
\]
\[
C=w_3F+w_4\widetilde C_4+w_5\widetilde C_5,\qquad E=C-F
\]

P3自身参与竞争。当w3趋于1时，语义修正E趋于0，而不是强制向P3注入粗尺度上下文。
这些权重不是经GT监督或校准的“目标可靠性概率”。

### 4. 细节与语义修正双门控及交互

\[
(G_D,G_C)=Sigmoid\left(h_G([F,|E|,|D|])\right),\quad
U_D=G_D\odot D,\quad U_C=G_C\odot E
\]
\[
P'_3=P_3+\beta\odot W_o\left(\psi([U_D,U_C,U_D\odot U_C])\right),\quad
\beta_c=0.5\tanh(a_c)
\]

beta按通道独立学习，初始0.1、绝对值始终小于0.5；restore输出投影初始化为0。
初始预测精确等于基线，先更新输出投影，其后内部对齐/门控获得非零梯度。
不能同时把beta和restore都设为0；代码已禁止此双零初始化。

完整模块新增162,377参数；按原部署/日志口径预计总计10,340,578（10.341M）。
训练用GroupNorm的新增分支做activation checkpoint，原Backbone/FPN/PAN BatchNorm不重计算，
不会因checkpoint多更新一次running statistics。本地CPU验证不能保证4090峰值显存/速度，需真实运行测量。
部署检查包含同输入SADF分支与Decoder的数值一致性、整机forward有限输出；
未训练随机检测器的TopK并列候选可因原Backbone BN折叠的小数差异重排，不将其当成新增模块的部署误差。
没有验证ONNX/TensorRT导出。

## 与已有方法的边界

- [FaPN，ICCV2021](https://arxiv.org/abs/2108.07058)：学习跨层对齐与特征选择。SADF借鉴对齐问题意识，
  不是FaPN官方实现；这里以浅层对比细节参与P4/P5的有界单点流场，并保留原D-FINE融合路径。
- [ASFF](https://arxiv.org/abs/1911.09516)：学习空间尺度权重以抑制跨尺度冲突。SADF借鉴尺度竞争，
  自尺度参与、对齐后的差异以及显式的context_delta都是本次要核对的机制选择。
- [Focal Modulation，NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1b08f585b0171b74d1401a5195e986f1-Abstract-Conference.html)：
  多范围上下文、门控聚合与调制。SADF不替换AIFI，也不是将FocalNet改名。

拟研究的整体贡献是“**局部对比引导对齐、对齐差异引导尺度竞争、双门控约束细节与上下文的注入**”。
三个组成思想都不是凭空全新。能否形成独立创新点，还要检索更近方法、比较及验证；未声称首创。

## 当前唯一训练配置

`configs/dfine/dfine_s_scbs_hrw_110_SADF.yml`只写本实验启用项，继承现有服务器基线。
SCB-S原始标注；0=hand-raising、1=read、2=write；110轮、100停止增强、batch32、seed0、原优化器/损失。
从头训练，保留原HGNetV2预训练初始化；不使用旧检测器权重微调作为正式可比结果。

训练前关注read/write AP、APm、尤其medium write AP能否一起提升，hand AP不要显著下降。
不预设必须提升多少，不靠APs判断（当前只有3个small GT）。

## 同步，再运行

本地Mac终端：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_SADF.sh
```

白名单同步修改源码和本实验文件，不覆盖服务器基线配置、数据、旧结果；被替换源码留备份。
不要从含未完成其他文献模块的暂存工作区分发。没有实际联网同步或启动训练，除非用户另外授权执行。

服务器：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_SADF_110.sh
```

脚本先核对单变量配置/数据并做合成测试；从头训练，实时tee日志；最佳stg2 checkpoint导出
model/EMA参数、query与混淆矩阵，再生成类别×尺度、候选缺口、score/PR分析。
原结果目录存在就停止，端口由torchrun standalone自动选择，不杀旧进程。
请确认GPU空闲；脚本没有替你停止任何其他任务。

默认输出：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_SADF_seed0_run1`

只演练（不要求GPU/数据，不训练）：

```bash
bash tools/diagnostics/run_scbs_SADF_110.sh --dry-run
```

## 后续支撑创新点的验证（本次不启动多组）

1. 完整SADF至少补重复实验，整体AP对照110轮三次均值；详细诊断固定Baseline Run2。
2. 接口`sadf_enable_alignment=false`：offset效果取消，判断对齐是否必要。
3. 接口`sadf_enable_detail=false`：对比细节通路取消，判断细节引导是否必要。
4. 接口`sadf_enable_scale_selection=false`：等权尺度融合，判断差异感知竞争是否必要。
5. 同参数量普通Conv/MLP融合、EncoderCrossGate、可比FaPN/ASFF适配对照，排除纯参数增量。
6. `encoder.sadf(source,p3,p4,p5,return_aux=True)`可输出detail/scale权重、双门控、实际offset场；
   仅分析/可视化时打开，不在训练时缓存计算图。按GT类别×尺寸汇总这些图可辅助解释，但仍不是因果证明。

消融接口保留同样参数/计算图（被取消分支梯度为0而非unused），为机制比较设计，
不是已经测过的高效简化模型。beta/offset预测器权重导出也不能替代实际采样图观察。
没有改原损失，不含用于“强制写字增多”的类别偏置。
