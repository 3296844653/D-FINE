# 论文分区式D-FINE-S + AQS架构图

布局参考：

- 用户提供的 `ELSEVIER HF-D-FINE.pdf` 第4页Figure 3：编码器分区、特征金字塔、token列。
- 用户提供的 `DETRPose(1).pdf` 第6页Figure 3：query选择、Decoder层级及各层输出分支。

只参考图的呈现方式，不使用参考论文图像作为背景，不移植其模块或任务。
没有HF Hybrid Encoder、额外P2、CAF、UCS/DCS、Outer-SNWD、Pose LQE、关键点分支或Group Self-Attention。

## 文件

- `dfine_s_aqs_publication_style.drawio`：所有框、特征平面、token符号、文字、连接线都是原生可编辑元素。
- `dfine_s_aqs_publication_style.svg`：纯矢量，无论文截图或内嵌图片。
- `dfine_s_aqs_publication_style.png`：高分辨率预览。
- 工程 `output/pdf/dfine_s_aqs_publication_style.pdf`：纯矢量PDF，原生文字可提取。
- `generate_publication_figure.py`：生成程序，复用 `../aqs_l2_r1/generate_diagrams.py` 的Scene基础。

## 与实际代码的对应

当前配置为 `configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`：
HGNetV2-B0、原HybridEncoder、640输入、3个特征尺度、300个普通query、3层Decoder、AQS仅在第2层后。

- 原骨干返回P3/P4/P5，分别为256×80×80、512×40×40、1024×20×20；省略batch维。
- 1×1投影统一到256通道，仅P5通过配置的Transformer/AIFI。
- 编码器展示原FPN top-down和PAN bottom-up融合。 lateral Conv/Norm等算子为结构级折叠；不是逐算子计算图。
- token列用示意符号，不是逐个画满8400个memory token或300个query；原Top300规则不变。
- 蓝色共享memory路径连到三个Decoder层的MSCA，区别于query初始化路径；AQS没有直接memory输入。
- SA/MSCA/FFN分别表示self-attention、多尺度deformable cross-attention、feed-forward network。
- 传统bbox分支从L1生成b0；FDR头预测残差corner logits并层间累加z，再softmax得到Pr。最终边偏移按W(n)积分；各层的完整框恢复及下一层reference-position routing未逐算子展开。
- L2 query以及原ClsHead2+LQE2得到的logits进入AQS。L2 FDR在AQS前计算，AQS不直接输入或修改当前bbox。
- 增强query进入L3，最终分类与定位都可能间接受影响。原损失/GO-LSD保留。
- 推理图省略DN和训练辅助监督；训练时DN和普通query分别计算AQS，不互相池化。

全部特征平面与token符号为结构示意，不是模型实测activation/attention，不展示预测例子或虚构性能。
原始参考论文、先前架构图与训练代码均未修改，当前服务器实验不受影响。

建议图注：D-FINE-S加入AQS后的整体结构。保留原有HGNetV2骨干、Hybrid Encoder和FDR定位头，
在第二层Decoder之后引入基于分类置信度的门控上下文增强，残差更新query并传递至第三层Decoder。
编码器与token/Decoder布局分别参考HF-D-FINE Figure 3和DETRPose Figure 3的画法，模型结构以本地代码为准。
