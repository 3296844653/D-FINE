"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

"""
表示从当前目录的 dfine.py 文件中导入 DFINE 类。
其中：. 表示当前 Python 包目录；
    dfine 对应 dfine.py；
DFINE 是文件中定义的主模型类。
DFINE 一般负责把模型的几个主要部分组织起来，例如：
    输入图像
    ↓
    Backbone 骨干网络
    ↓
    HybridEncoder 编码器
    ↓
    DFINETransformer 解码器
    ↓
    分类与边界框预测
也就是说，DFINE 通常是训练和推理时直接调用的完整模型入口。
"""
from .dfine import DFINE
"""
它主要负责计算模型训练过程中的损失，例如可能包括：
    分类损失；
    边界框 L1 损失；
    GIoU 损失；
    D-FINE 的分布式边界框回归损失；
    辅助解码层损失；
    去噪训练损失；
    一对多监督损失。
简单理解：
模型预测结果 + 真实标签
             ↓
    DFINECriterion
             ↓
    总损失 loss
             ↓
    反向传播更新参数
它只在训练阶段发挥主要作用，模型推理时通常不需要计算这些损失。
"""
from .dfine_criterion import DFINECriterion
"""
这是 D-FINE 最核心的模块之一，主要负责：
    接收编码器输出的图像特征 memory；
    选择初始目标查询；
    初始化参考框；
    让目标查询与图像特征交互；
    逐层更新分类结果和预测框；
    输出最终的类别分数和边界框。
整体流程可以理解为：
    编码器特征 memory
            +
    目标查询 query
            +
    初始参考框 reference points
            ↓
    DFINETransformer
            ↓
    每一层解码器的分类结果和边界框结果
"""
from .dfine_decoder import DFINETransformer
"""
它负责处理骨干网络输出的多尺度特征图。
例如，骨干网络可能输出：
    P3：80 × 80，高分辨率特征
    P4：40 × 40，中等分辨率特征
    P5：20 × 20，低分辨率特征

HybridEncoder 会对这些特征进行：
    通道维度统一；
    特征投影；
    Transformer 编码；
    多尺度特征融合；
    FPN/PAN 风格的自顶向下和自底向上融合；
    输出供解码器使用的多尺度特征。
大体流程是：
    Backbone 多尺度特征
            ↓
    通道统一与特征投影
            ↓
    部分尺度进行 Transformer 编码
            ↓
    多尺度特征融合
            ↓
    提供给 DFINETransformer
"""
from .hybrid_encoder import HybridEncoder
"""
DETR 系列模型会产生固定数量的预测，例如 300 个预测框，但一张图片中的真实目标数量可能只有十几个。
因此，训练时需要解决一个问题：
    哪一个预测框对应哪一个真实目标？
HungarianMatcher 使用匈牙利算法，在模型预测和真实标签之间进行一对一匹配。
匹配代价通常由以下部分组成：
    匹配代价 =
    分类代价
    + 边界框 L1 距离代价
    + GIoU 代价

例如有三个真实目标和多个预测：
真实目标 A ←→ 预测 Query 18
真实目标 B ←→ 预测 Query 76
真实目标 C ←→ 预测 Query 203
匹配完成后，DFINECriterion 才能知道应该用哪个预测结果计算哪个真实目标的损失。

因此二者的关系是：
    HungarianMatcher：确定谁和谁匹配
    DFINECriterion：对匹配后的结果计算损失
"""
from .matcher import HungarianMatcher
"""
模型原始输出通常不能直接用于展示或评估，例如：

    分类输出可能还是 logits；
    边界框可能是归一化的 cx, cy, w, h；
    坐标范围可能在 0～1；
    还需要选出得分最高的类别和预测框。

DFINEPostProcessor 会把模型输出转换成最终检测结果。

例如：

模型原始输出：
类别 logits
归一化边界框 [cx, cy, w, h]

        ↓ DFINEPostProcessor

最终输出：
类别标签 labels
置信度 scores
像素坐标边界框 [x1, y1, x2, y2]

它主要用于：
    验证阶段；
    COCO 指标计算；
    实际推理；
    检测结果可视化。
"""
from .postprocessor import DFINEPostProcessor
