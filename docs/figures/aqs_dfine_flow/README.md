# D-FINE-S + AQS-L2：依据源码的架构图

用户参考图仅用于布局：输入图像与下方 Backbone、灰色 Hybrid Encoder、绿色特征层、AIFI 的特征与位置编码、浅绿色跨尺度融合区域、竖向 Query Selection、Decoder 与检测输出。
本图是纯原生矢量图，`.drawio` 为编辑源文件，另有 SVG/PDF/PNG；没有使用参考图截图作为背景。两侧课堂图标都是示意，不能用作真实预测或效果证据。
只创建图形文档；未修改训练代码、配置、权重、数据或服务器。

## 核对依据

- `configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`
- `configs/dfine/dfine_hgnetv2_s_coco.yml`：HGNetV2-B0、三层 Decoder。
- `configs/dfine/include/dfine_hgnetv2.yml`：三尺度、P5 注意力、Top300、默认 Query 选择。
- `src/zoo/dfine/hybrid_encoder.py`：1x1 投影、P5 AIFI、两个 top-down FPN 融合、两个 bottom-up PAN 融合，实际使用 RepNCSPELAN4 与 SCDown。
- `src/zoo/dfine/dfine_decoder.py`：原候选选择、LQE、FDR、AQS 位置和输入。

## 图形抽象与实际流程

1. Backbone 的三尺度输出先经各自 1x1 投影；图中的 Projection 是整组三个投影的总览容器。
2. P5 特征展平后进入 AIFI，位置编码输入保留。实际位置编码加入 Transformer 的 Q/K，V 保持来自图像特征；图上以独立蓝色输入表示，而非修改图像特征本身。
3. AIFI 的 P5 输出与投影后的 P3/P4 进入原跨尺度融合。FPN/PAN 图框分别表示完整阶段，不是将四个融合算子改成两层。PAN 阶段的输出包括 FPN P3 直通与融合后的 P4/P5，因此完整金字塔用单条特征束表示。
4. Encoder memory 同时送入原 Query Selection 和三层 Decoder 的多尺度交叉注意力。蓝色支路是共享 memory，不直接进入 AQS。
5. 默认候选选择按 encoder 分类 logits 的最大类别值取 Top300；不是参考图所标的“Uncertainty-minimal Query Selection”，更不是新增 AQS 在此筛掉候选。Query content 与参考框初始化细节收在这一总览阶段中。
6. Decoder L2 的原 FDR 先计算分布/框，原 Cls2+LQE2 得到 AQS 的门控 logits。L2 query 与该 logits 分别进入 AQS；更新后的 query 送入 L3。
7. AQS 不删除 query，不改变候选数，不替换原损失；L2 当前 FDR/框不经 AQS。L3 分类与定位都可能间接受 query 更新影响。
8. FDR2 -> FDR3 虚线表示原累积分布残差状态。最终 FDR3 框包含 Integral/distance2bbox 的原解码过程，最终 Cls3+LQE3 输出分数。
9. L1 的初始 bbox 头、逐层参考框/位置初始化、L1 分布残差、DN 与训练辅助监督未逐项展开；这些路径并未从实现中删除。

## 使用

draw.io 打开 `dfine_s_aqs_flow.drawio` 即可编辑。Encoder、Decoder、token 列和课堂图标已分组；可整体移动，也可进入组修改单个原生元素。
源码生成器会重写本目录的同名文件；若手工调整过布局，请另存 `.drawio` 后再运行生成器。
生成器自动核验关键源码、AQS-L2 路径、单条 FPN/PAN 连接、所有线段水平/垂直、原生 XML 和无位图 PDF。
