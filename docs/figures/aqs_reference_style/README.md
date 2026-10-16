# D-FINE-S + AQS：用户参考图版式

文件 `dfine_s_aqs_reference_style.drawio` 是编辑源文件；SVG、PDF 为矢量导出，PNG 用于预览。
仅修改图形文档，没有修改训练代码、配置、数据、权重，也没有同步服务器。

## 当前手工编辑版

当前 `.drawio` 已由用户替换输入/输出照片、调整输出布局和 AQS 配色；以该文件为准。
本次仅补回输出 token 到 hand/read/write 三张图片的三条颜色对应曲线，其他已有图形、图片与样式原样保留。
**不要运行下面的历史生成器来覆盖当前文件**，它们会重建旧示意图并丢失手工修改。
当前版本的 PDF、SVG 和 PNG 使用本机 draw.io 原生导出，照片保留，文字和连线仍为矢量。
手工原件备份：`versions/dfine_s_aqs_user_before_output_repair_20261014.drawio`。

## 版式

- 按用户所给图片采用横向总览、浅灰圆角 Encoder 分区、绿色堆叠特征图、竖向 token/Query selection/Decoder。
- 按后续要求取消深色创新强调区域与阴影。AQS 使用普通浅色方块，只标明位置。
- 去掉模块内的小字副标题和旁边的操作说明，只保留模块名称、特征层与 Top300 等必要标记；详细机制保留在本说明中。
- 最新版按用户完整参考图重画 Encoder：AIFI 上方补输出特征方块与网格特征图，下方补蓝色位置编码和绿色图像特征方块，三尺度特征图改成紧密重叠排列。
- CCFF 内采用单列三尺度 Fusion 示意，旁边用黄色/蓝色操作符概括 top-down/bottom-up 两个方向；两条蓝色 lateral 旁路按参考图回折，交叉处用跨线桥表示不相连。C 移入 CCFF，三尺度输出分别从上、左、下汇入 C。
- 三个 Fusion 框是三个输出尺度的**示意容器**，不是源码里的三层 Fusion。D-FINE 实际两次 FPN、两次 PAN 保持不变：P3 对应 FPN 最细尺度输出，P4 行封装 top-down 和 bottom-up 的两个不同状态，P5 对应最终 PAN 粗尺度输出。因此这是尺度级交互示意，不应把图上双向路径理解成模型新增循环或共享 Fusion 参数。
- 黄色方块概括原 lateral 1x1/上采样路径；蓝色方块概括原 SCDown 路径，不代表照搬参考图的普通 Conv3x3+BN+SiLU，也没有新增操作。
- 最新整图版使用1842x620画布：输入图片在上方，向下收窄的浅黄色 Backbone 在下方；右侧使用细长 token 列、黄色 Query selection 和紧凑蓝色 Decoder & Head。
- 按最新圈选批注，AIFI、上下特征方块、A5 网格和 P3/P4/P5 堆叠图整体等比例缩至80%，字体与节点尺寸一起缩放。CCFF、Encoder 总标题、输入/Backbone、AQS 和右侧流程均不缩放；只重新对齐连接所需的接口连线，P3 到 Fusion 仍保持水平直线。
- 随后的两条批注进一步要求横向收窄：整体 Encoder 横向压缩20%，CCFF 区域再额外压缩15%，最终 Encoder 宽度640 -> 482.12、CCFF宽度249 -> 169.32。整体高度和文字大小保持不变，操作符方块与 C 保持方形/圆形。
- 为避免编码器右侧出现长空白，将完整 Memory/Query/Decoder/AQS/输出/图例区域整体左移158单位，所有内容、大小、层次与连接关系不变；画布随之变窄。标题仅重新居中。
- 最新输入区域批注：输入图像和 Backbone 整体上移32.42单位，使 Backbone 的实际下沿出口与 P3 入口同高，取消全部出口折点，改为水平直箭头。Input image 标签置于图像上方，避免竖向箭头穿过文字；Encoder 和右侧区域不变。
- 随后按左侧区域批注，将 AIFI、特征方块、A5 网格与 P3/P4/P5 整体放大12%，字体一同放大，并将该组合的竖向中心对齐 CCFF 区域中心。CCFF、输入/Backbone 和右侧模块均不移动、不缩放；P3 入/出口重新附着到特征图实际斜边，仍保持水平直线。
- Decoder 内明确显示 Layer 1 -> Layer 2 -> AQS -> Layer 3 -> Head。颜色和整体布局参照用户新图，不再冻结上一版右侧位置；模型流程不变。
- 右上操作符图例按 D-FINE 调整：黄色概括 lateral 1x1/Norm/Act 与上采样，蓝色为 SCDown（1x1 Conv -> depthwise 3x3 stride2），不能直接写成参考图的普通 Conv3x3。
- 输出端添加三条颜色对应曲线，右下添加位置编码、图像特征、Object Query 图例。颜色曲线仅为示意对应，不是增加检测分支，也不表示实测预测。
- 主干数据流路径为水平或垂直；蓝色旁路交叉处使用跨线桥，特征图内部网格按平行四边形倾斜，输出颜色对应线为曲线。这只是总览图的简化，没有改动模型。
- 未添加参考论文的 UCS/DCS/CAF、额外 P2 或关键点分支。
- 两侧课堂图标均为原生矢量示意，不是真实输入样本或实测预测，不能用作效果证据。

## 与当前实现对应

- 配置依据：`configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`。
- HGNetV2-B0 -> 三层 P3/P4/P5 -> 1x1 投影至256通道 -> P5 AIFI -> 原有两次 FPN 与两次 PAN。
- AIFI 两侧绿色方块表示图像特征，蓝色表示 Q/K 的位置编码示意，不是新增网络层；实际 V 不加位置编码。
- 黄色路径概括 FPN 的 lateral Conv、上采样、Concat/CSP；蓝色竖向路径概括 PAN 的下采样、Concat/CSP，回折路径概括原 lateral skip。黄色方块隐藏了卷积与上采样两个步骤，回折旁路实际取上采样前的 lateral 特征。
- FPN 的 P3 输出与 PAN 的 P4/P5 输出构成共享多尺度 memory；图中三条输出汇入 CCFF 内的 C 后送入 Memory。这不是新增特征融合算子。
- 三尺度 Fusion 容器不逐个列出底层 RepNCSPELAN4 算子；输入 Projection、Concat 内部操作和训练路径仍在总览中省略，没有从模型中删除。训练实现与权重没有任何变化。
- 原有 encoder 候选选择保留 Top300 普通 query。图中的 token 方块数量仅作示意。
- AQS 放在 Decoder L2 后、L3 前。依据 L2 的 query 与 ClsHead/LQE 后 logits 做门控、上下文池化、残差更新。
- AQS 不删除 query，不替换原有 query 初始化，也不修改原损失。
- 图为总览，省略逐层 FDR、参考框、LQE 质量输入、DN 与训练辅助监督。L2 的 FDR/框是在 AQS 前计算；AQS 更新后的 query 继续送入 L3，故后续分类与定位均可能间接受影响。

## 编辑

在 draw.io 中打开 `.drawio`。文字与图形都是原生元素；Encoder、Decoder、token列、课堂图标已经分组，可整体移动。
要改内部单个元素，可以双击进入组或右键取消组合。连线绑定对应节点，不是位图上的画线。
请保留 `.drawio` 源文件，论文排版时导出 SVG/PDF。

## 再生成

运行 `generate_compact_flow.py`（需要 reportlab、pypdf、pypdfium2，以及 macOS Times New Roman 字体）。`generate_reference_figure.py` 保留为上一版编码器布局与原生矢量组件库。

上一版已保留在 `versions/encoder_only_20261014/`，可恢复。所有版本均只改图形文档，不改训练配置或代码。
本次缩小前的完整整图保留在 `versions/full_flow_before_aifi_resize_20261014/`。
本次横向收窄前的版本保留在 `versions/before_encoder_narrow_20261014/`。
输入区域上移前的版本保留在 `versions/before_input_raise_20261014/`。
本次左侧区域放大前的版本保留在 `versions/before_aifi_enlarge_20261014/`。
此生成器会重写本目录同名导出文件，不会修改其他版本；在 draw.io 中做了手工编辑后，应另存源文件，避免再次运行覆盖手工布局。
