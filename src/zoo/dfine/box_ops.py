"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

"""
这个 box_ops.py 是 目标检测中的边界框工具文件，来源于 DETR。它不包含网络层，也不会直接提取特征，主要负责：

    边界框格式转换；
    计算 IoU；
    计算 GIoU；
    将分割掩码转换成边界框。

这些函数会被 D-FINE 的匹配器、损失函数和后处理模块调用。
"""


import torch # 进行张量计算
from torch import Tensor # 主要用于类型标注
from torchvision.ops.boxes import box_area # 导入 torchvision 已经实现好的边界框面积计算函数 要求输入框格式为： [x_min, y_min, x_max, y_max] 面积计算方式为：area=(xmax−xmin)(ymax−ymin)

"""
边界框的两种表示方法
1. cxcywh 格式
    [cx, cy, w, h]
其中：
    cx 边界框中心点横坐标
    cy 边界框中心点纵坐标
    w 边界框宽度
    h 边界框高度

例如：
    [50, 40, 20, 10]
表示：
    中心点为 (50, 40)
    宽度为 20
    高度为 10
    
2. xyxy 格式
    [x0, y0, x1, y1]
也可以写成：
    [x_min, y_min, x_max, y_max]
其中：
    (x0, y0)：左上角
    (x1, y1)：右下角
刚才的框转换后是：
    [40, 35, 60, 45]
"""


"""
把边界框从：
    [cx, cy, w, h]
转换成：
    [x0, y0, x1, y1]
也就是从“中心点加宽高”转换为“左上角和右下角”。
1. 拆分最后一个维度
x_c, y_c, w, h = x.unbind(-1)

假设输入形状是：
    [N, 4]
例如：
    x = tensor([
        [50, 40, 20, 10],
        [30, 60, 10, 20]
    ])
执行 unbind(-1) 后：
    x_c = [50, 30]
    y_c = [40, 60]
    w   = [20, 10]
    h   = [10, 20]
-1 表示最后一个维度。
2. 计算左上角和右下角
    x_c - 0.5 * w 得到左边界：x0=cx−2w
    y_c - 0.5 * h 得到上边界：y0=cy−2h
    x_c + 0.5 * w 得到右边界：x1=cx+2w
    y_c + 0.5 * h 得到下边界：y1=cy+2h
3. 为什么使用 clamp
    w.clamp(min=0.0)
    h.clamp(min=0.0)
clamp(min=0.0) 会把负数限制为 0。
例如：[-2, 5, 10]处理后变成：[0, 5, 10]因为边界框宽度和高度理论上不能为负数。
如果模型意外预测：w = -0.2 经过处理后： w = 0  可以避免生成左右位置颠倒的非法边界框。
4. 重新组合
return torch.stack(b, dim=-1)
把四个坐标重新组合成：
    [x0, y0, x1, y1]
输出形状与输入基本相同，最后一维仍然是 4。
例如：[50, 40, 20, 10]转换为：[40, 35, 60, 45]
在 D-FINE 中的用途

D-FINE 模型内部经常使用：
    [cx, cy, w, h]
表示预测框。
但是计算 IoU、GIoU 时，需要转换为：
    [x0, y0, x1, y1]
因此损失函数中通常会出现类似代码：
    generalized_box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes)
    )
"""
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)

# 把边界框从:[x0, y0, x1, y1] 转换成： [cx, cy, w, h] 它与前一个函数正好相反
def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
# 计算两组边界框之间的两两 IoU，同时返回并集面积
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 计算交集区域的左上角
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # 计算交集区域的右下角
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # 计算交集面积
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # 计算并集面积
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

# 广义 IoU 计算
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # 检查框是否有效
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 计算最小闭合包围盒的左上角
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # 计算最小闭合包围盒的右下角
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # 计算最小闭合包围盒的面积
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

# 将掩码转换为边界框
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # 如果掩码为空，返回空张量
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    # 提取掩码的高和宽
    h, w = masks.shape[-2:]

    # 创建坐标网格
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    
    # 计算掩码的坐标值
    x_mask = masks * x.unsqueeze(0)
    # 计算掩码的最大坐标
    x_max = x_mask.flatten(1).max(-1)[0]
    # 计算掩码的最小坐标
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # 计算掩码的坐标值
    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
