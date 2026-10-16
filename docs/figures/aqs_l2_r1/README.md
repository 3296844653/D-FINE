# D-FINE-S + AQS-L2 R1 架构图

本目录只包含文档图，不修改模型源码、训练配置或正在运行的实验；无需同步服务器。

## 文件

- `dfine_s_aqs_l2.drawio`：原生可编辑文件，含整体架构与AQS内部结构两页。
- `dfine_s_aqs_l2_architecture.svg`：整体架构矢量图。
- `aqs_l2_module.svg`：AQS内部结构矢量图。
- 同名PNG：高分辨率预览；论文排版优先SVG或从draw.io导出PDF。
- `generate_diagrams.py`：从同一套场景定义生成SVG和原生draw.io节点与连线。

用draw.io打开 `.drawio` 文件，各框、文字、连接线均可单独编辑；不是整张SVG图片。
图中不含人工挑选的检测示例或未经验证的性能数字。最终论文排版前可统一字体。

## 源码依据

- `configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`
- `configs/dfine/dfine_hgnetv2_s_coco.yml`
- `configs/dfine/include/dfine_hgnetv2.yml`
- `src/zoo/dfine/dfine_decoder.py` 的AQS、分组封装与Decoder前向。

SCB-S三类D-FINE-S/HGNetV2-B0，普通query数300、hidden_dim256、Decoder三层，AQS仅在L2后。
输入640×640对应骨干P3/P4/P5为256×80×80、512×40×40、1024×20×20（省略batch维）。
编码器投影至256通道，仅P5执行配置的Transformer/AIFI，随后FPN/PAN跨尺度融合。
原有输出头、FDR框迭代路径是结构级概括，不是逐算子计算图。

## 关键实现说明

1. 原D-FINE Top300候选初始化不变。AQS不是另一个TopK选择器，不删除普通query。
2. 输入是L2 query和L2分类头经LQE后的logits；不是仅校准最终分数。
3. 重要性为最大sigmoid类别分数；前向硬门控、反向STE学习阈值。
4. 池化选中的原始query；MLP输入为LayerNorm(query)和广播的context。
5. 门控同时控制池化与残差写入；未选中query沿恒等路径保留。
6. tau初值0.50、原始rho初值0.10均可学习；T=0.10固定；有效残差为tanh(rho)。
7. MLP末层零初始化。L2 bbox/FDR在AQS前计算，不直接经AQS；更新query送入L3，L3分类与框均可能变化。
8. 图省略DN与训练辅助监督。训练时DN和普通query分别应用同一AQS，不相互池化；不是DN完全不经过AQS。
9. 不含BoxLocal、SERF、新损失；不宣称官方论文模块完整复现。

## 建议图注

整体图：添加AQS的D-FINE-S课堂行为检测网络。AQS插入第二层Decoder之后，根据分类置信度构建门控上下文，残差更新query并传递至第三层Decoder；骨干、编码器、候选数量和原有框迭代路径不变。

模块图：AQS内部计算。可学习阈值通过硬前向门控与直通估计器选择上下文query，池化context与query的归一化特征融合，再经门控和可学习有界残差系数写回query。

重新生成矢量文件：`python3 docs/figures/aqs_l2_r1/generate_diagrams.py`。
