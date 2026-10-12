# SDRF：语义差分关系融合（创新候选，不是已证明的创新点）

本次只修改 `/Users/wq/Desktop/D-FINE`，不修改SCB-U、标注或既有实验结果。
工作流程固定为：修改并验证桌面的本地项目 → 同步到 `/home/a5/MyProject/wq_project/D-FINE` → 在服务器运行。
同步脚本显式使用桌面项目的绝对源路径；不是从Codex worktree同步，也不会直接在服务器修改源码。
配置 `dfine_s_scbs_hrw_110_SDRF.yml` 独立继承110轮基线：batch32、停止增强100轮、seed0、原优化器/损失/EMA、原Top300候选；不叠加AQS或BoxLocal。

## 证据与待验证假设

110轮Baseline Run2诊断中，write→read为250/684（36.55%），read→write为168/2433（6.91%）。
684个write GT均存在IoU≥0.5候选，但214个没有正确类别的高分候选（置信度0.5诊断口径，非COCO AP的漏检定义）。
APm=41.14，APl=56.96；中尺度write AP=32.50，只有57个中尺度write GT，不能忽视其统计波动。
这些结果支持“有框时，局部判别信息不足”的研究假设，**并未证明特定Encoder算子是原因**。

AQS-L2 r0.1整体AP=55.87、write AP=49.74，但write→read仍有240个。
BoxLocal L2有局部query内容收益；EncoderCrossGate APm给出正信号而整体AP基本持平。
SADF、RWLocalDetail等已有尝试说明：简单地再融合更多细节或粗语义不能保证消除混淆。
本次研究问题是：**相似课堂行为的共同外观，能否通过细节位置间的语义条件关系差异得到更有区分力的表达？**

## 放置位置及具体机制

Backbone → 原Encoder投影/Transformer → 原FPN P4→P3融合 → **SDRF** → 原PAN → Decoder。

- `F3`：原始通道投影P3；细节query、key和value都来自这里。
- `S4`：FPN P4 lateral输出；投影、双线性上采样到P3坐标，只用于条件query。
- `Y3`：原FPN融合后的P3；仅作为保留基线路径的残差输入。
- SDRF直接更新FPN P3；随后PAN使变化传入P4/P5。因此它可能间接影响bbox和类别，**不是仅改最终分类，也不能声称P4/P5输出不变**。

在每个5×5 query块内，keys/values来自带1像素halo的7×7邻域。没有循环回绕或全图注意力；边界padding被显式屏蔽。
令 `Qf,Kf,Vf` 来自P3，`Qs`来自上采样P4，每头 `γ=sigmoid(g)`：

```
Lf = Qf Kfᵀ / sqrt(d) + relative_position_bias
Ls = Qs Kfᵀ / sqrt(d)
Af = softmax(mask(Lf))
As = softmax(mask(Lf + γ Ls))
D  = (As - Af) Vf
Y3_new = Y3 + α Wout SiLU(Whidden LayerNorm(D))
α = 0.5 tanh(raw_alpha)
```

两个注意力分布之差的每行和为零：共同的value分量可在关系消息中抵消，新增条件改变的细节关联则被保留。
这是算子性质，**不等于已经证明真实图像噪声被去掉**。没有手/笔/桌面监督，也不能称其已经学到了这些解剖或物体交互。
P4不进入value流，但条件仍可能有害；需实际消融和可视化验证。

α初值0.10；γ每头初值0.5；最后输出卷积零初始化，故训练起始输出与基线一致。
独立RNG构造保留原网络参数初值。非零α避免残差系数和输出投影同时为零造成分支学不动。
GN/LN无运行统计；新分支默认启用梯度检查点，降低反向保存开销而不重复原FPN/PAN的BN更新。
FP32 softmax和注意力积用于AMP数值稳定。显存/速度仍需在4090实际测量，不能承诺无额外训练成本。

新增参数70,377个（约0.070M）；本地原始模型合计10,293,386个，deploy口径为10,248,578个。
与论文表格比较必须统一原始/deploy统计口径，不能把两者直接相减。

## 文献依据及与已有模块的区别

1. [Differential Transformer, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)：借鉴两个softmax注意力图相减的思想。原论文以语言建模为主、使用双Q/K及学习差分系数；本模块不是其原样复现，而是细尺度共享K/V、粗尺度条件query、固定单位差分和有界外层残差。
2. [Scaling Local Self-Attention for Parameter Efficient Visual Backbones (HaloNet), CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Vaswani_Scaling_Local_Self-Attention_for_Parameter_Efficient_Visual_Backbones_CVPR_2021_paper.html)：借鉴分块query及周边halo扩大局部关联范围，不依赖新CUDA扩展。

相关工作检索还发现[DiffCLIP](https://arxiv.org/html/2503.06626v1)已把差分注意力用于视觉/文本编码器；
[D3FNet, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/DriveX/papers/Liu_D3FNet_A_Differential_Attention_Fusion_Network_for_Fine-Grained_Road_Structure_ICCVW_2025_paper.pdf)研究差分注意力与细粒度道路结构提取。
因此不能把“首次在视觉任务使用差分注意力”或“加入差分注意力”本身写作创新。
本项目可检验的贡献候选是：**粗语义仅改细节关联、共享细节K/V、条件前后关系差分、在D-FINE融合前恢复行为判别信息**这一具体设计及其实验论证；尚未完成全领域重复设计排查。

| 已有方法 | 本次核心差异 |
| --- | --- |
| EncoderCrossGate / DetailOnlyCrossGate | 主要是同位置门控或对比细节融合；SDRF改变多个位置之间的有符号关系 |
| SADF | 多尺度对齐、竞争和细节/上下文门控；SDRF不做偏移对齐，不把粗语义注入value |
| WindowInteractionP3 | 普通P3局部自注意力；SDRF加入P4条件、halo及两路关系差分 |
| AQS | Decoder query选择/context残差；SDRF在Encoder特征格点建模，不用类别分数选择query |
| BoxLocalQueryAdapter | 预测框内query取样；SDRF不需要预测框，发生在query初始化前 |

以上是本项目代码层面的区别，不是全领域新颖性证明。要成为论文创新点，仍需系统相关工作排查、机制消融、稳定收益和复杂度/失败情况报告。

## 必须能区分的机制消融

当前仅写主实验配置，不提前铺开多组训练配置。以后在独立配置的`HybridEncoder`中控制：

| semantic_condition | differential | 用途 |
| --- | --- | --- |
| True（默认） | True（默认） | SDRF完整机制 |
| True | False | 语义条件halo注意力，无差分 |
| False | False | 仅细节halo注意力 |
| False | True | 恒等关系修正的实现sanity check，不是有用的增强分支 |

对应配置名：`sdrf_semantic_condition`、`sdrf_differential`。保留相同参数形状，但关闭条件会有零有效梯度，因此应同时说明有效参数/算量区别。
固定设置存为整型buffer，严格恢复会拒绝不一致配置，避免评估时静默变换模块；消融从头训练，不能直接用不匹配配置恢复主实验。
独立主实验有效后再试与AQS结合，当前有互斥检查，避免不清楚收益来自哪里。

## 验证指标和判定

诊断统一与**110 Baseline Run2**比较；总体性能同时参考三次基线均值55.1565及其波动。

- 同一整体AP最佳epoch（best_stg2 EMA）比较AP/AP75、read AP、write AP、APm以及中尺度read/write AP。
- 相同val、confidence=0.5、IoU=0.5比较write→read和read→write，二者都列出；不能只提高write分数，把read错分成write。
- 正确高分候选缺口、类别×尺度覆盖和best-IoU query的分类分数；不能只看最佳IoU已很高。
- PR曲线、不同阈值及高分未匹配预测，避免混淆减少只是阈值/分数分布变化。
- 未匹配预测可能是重复、定位、漏标等，不一概称背景。
- 中尺度write只有57个、小目标总数仅3个：列数量、逐例观察，不用APs作为主要结论。
- 首轮只是筛选；重复训练、消融和参数/显存/时间测量后才讨论稳定性及创新贡献。

训练无需下载新预训练模型给SDRF，仍使用已有基线HGNetV2权重；第一次本地合成测试关闭预训练避免网络影响。

## 使用

Mac先同步：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_SDRF.sh
```

服务器再运行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_SDRF_110.sh
```

脚本沿用`conda run --no-capture-output -n wq torchrun --master_port=7781 --nproc_per_node=1`，不改成standalone；实时终端日志同时存入console.log。
端口被占用可设置`SCBS_SDRF_PORT=7782`；不自动停止其他项目/GPU进程。
已有输出会停止；重新实验可传新的绝对SDRF输出目录。

输出：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_SDRF_seed0_run1`。
训练后自动导出最佳model/EMA参数、混淆矩阵、原query诊断及离线分析，分别位于`best_stg2_SDRF_parameters.json`、`diagnostics_eval/evaluation`、`diagnostics_eval/analysis`。
同步限定新增模块及两处入口代码；备份替换的服务器源码，不覆盖数据/基线配置/旧实验，也不自动启动训练。
