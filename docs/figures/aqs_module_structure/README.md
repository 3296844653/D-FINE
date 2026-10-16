# AQS 内部结构图

独立展示当前 `AdaptiveQuerySelectionRefiner`（legacy AQS-L2），不覆盖整体架构图或训练代码。

## 当前样式

按用户提供的竖向模块图借鉴视觉样式：浅粉底、虚线输入/输出分组、彩色 query 与二值门控、粉色自适应门控模块、灰色池化/融合框和黑色箭头。参考图的 **Adaptive Quantization Selection** 并不是本项目的 **Adaptive Query Selection**，因此不引入量化器、attention map 输入或额外正则化损失。

按最新要求去掉图中的公式、数字、参数、连线注释和页脚说明，只保留模块名称；主流程等宽等高，池化、融合和残差更新模块等距排列，输入和输出虚线区域对齐。

图中8个置信度、二值门控和 query 色块仅用于示意，并非真实预测或实验数据。它们保持逐位置对应，门控为0的query仍出现在输出；门控为1的输出query用短白条示意残差更新。真正的普通query数量保持300。图中把LayerNorm、拼接与MLP合并成Context Fusion框，精确步骤见下方。

旧横版和脚本保留在 `versions/horizontal_20261015/`，不修改已有整体架构图。
带公式的上一版竖图保留在 `versions/portrait_detailed_20261015/`。

- 编辑源文件：`aqs_module_structure.drawio`；同时导出 SVG、PNG 与 `output/pdf/aqs_module_structure.pdf`。
- 原始依据：`src/zoo/dfine/dfine_decoder.py` 的 `AdaptiveQuerySelectionRefiner.forward`、`TransformerDecoder._apply_aqs_refiner` 与 decoder L2 调用路径。
- 配置：`configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`；τ初值0.50，ρ初值0.10，固定T=0.10。
- 普通检测 query 为 `[B,300,256]`；图中省略batch维。S₂为当前 L2 原分类头经 LQE 修正后的类别 logits。

计算顺序：

1. `s_i = max_c sigmoid(S_i,c)`。
2. `τ = sigmoid(θ)`；`g_s = sigmoid((s_i-τ)/T)`；前向硬门控，训练用STE梯度：`g = stopgrad(h-g_s)+g_s`。
3. 从**原始query**聚合 `c = sum(g_i*q_i)/max(sum(g_i),1)`，再广播；不是用分类置信度做softmax注意力。没有query被选中时c为零。
4. 拼接 `[LayerNorm(q_i); c]`，经 `Linear(512,256) -> SiLU -> Linear(256,256)` 得到Δq_i。
5. `q_i' = q_i + tanh(ρ)*g_i*Δq_i`，保持全部300个query；门控同时控制池化和更新，并不删除query。
6. 更新后query送入L3。L2的框/角点分布在AQS之前计算，AQS没有直接的bbox或feature map输入。

训练时DN与普通query分开池化，避免GT信息混入普通query。MLP最后一层权重和偏置零初始化，因此初始输出为恒等映射。
图为普通query的推理主路径，省略公式与训练细节；不改原损失，也没有加入EvidenceContrastAQS等其他模块。

图形、文字和连接线均为原生矢量，可在draw.io编辑；再生成请运行本目录 `generate_aqs_module.py`。
