"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch # 用于创建和处理张量

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh # 边界框格式转换 添加边界框噪声时，代码先把框转换成 xyxy 格式，因为这样可以直接对上下左右四条边进行扰动
from .utils import inverse_sigmoid # 把归一化到 [0,1] 范围的边界框坐标转换到 Sigmoid 之前的数值空间 解码器内部通常保存的是未经过 Sigmoid 的参考框坐标，因此生成去噪框后要执行 inverse_sigmoid
 

"""
这个 denoising.py 实现的是 D-FINE 训练阶段的对比去噪查询构造。核心函数是：
    get_contrastive_denoising_training_group(...)
它利用真实标签和真实边界框，生成带噪声的类别查询和边界框查询，然后与正常检测 Query 一起送入 Transformer 解码器。
其目的不是清理图片噪声，而是让模型学习：
    即使类别和边界框受到一定扰动，也能把它们恢复成正确目标。
这种训练机制可以加快 DETR 类模型收敛，并提高 Query 学习和边界框定位的稳定性。
"""

"""
这个函数主要完成五件事：
    将一个 batch 中不同数量的真实目标补齐；
    复制真实目标，构造多组正负去噪 Query；
    对类别标签加入随机噪声；
    对真实边界框加入随机噪声；
    构造注意力掩码，避免不同去噪组相互泄漏答案。
"""
def get_contrastive_denoising_training_group(
    targets, # 一个 batch 的真实标注列表
    num_classes, # 类别数量
    num_queries, # 正常目标检测 Query 的数量
    class_embed, # 类别嵌入层
    num_denoising=100, # 期望构造的去噪样本规模
    label_noise_ratio=0.5, # 类别标签噪声比例
    box_noise_scale=1.0, # 边界框噪声强度
):
    """cnd"""
    # 关闭去噪训练时直接退出
    if num_denoising <= 0:
        return None, None, None, None

    # 统计一个 batch 中的真实目标数量
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    # 处理整个 batch 没有目标的情况
    if max_gt_num == 0:
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return None, None, None, dn_meta

    # 计算去噪组数
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    # 将不同数量的真实目标补齐
    bs = len(num_gts)

    # 类别张量
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    # 边界框张量
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    # 有效目标掩码
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    # 把真实标注写入补齐后的张量
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    # 复制成多组正负去噪 Query
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    # 区分正去噪和负去噪 Query
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # 保存正去噪 Query 的索引
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    # 计算实际去噪 Query 数量
    num_denoising = int(max_gt_num * 2 * num_group)

    # 加入类别标签噪声
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # 加入边界框噪声
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        # 随机决定扰动方向
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        # 随机决定扰动幅度
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        # shrink_mask = torch.zeros_like(rand_sign)
        # shrink_mask[:, :, :2] = (rand_sign[:, :, :2] == 1)  # rand_sign == 1 → (x1, y1) ↘ →  smaller bbox
        # shrink_mask[:, :, 2:] = (rand_sign[:, :, 2:] == -1)  # rand_sign == -1 →  (x2, y2) ↖ →  smaller bbox
        # mask = rand_part > (upper_bound / (upper_bound+1))
        # # this is to make sure the dn bbox can be reversed to the original bbox by dfine head.
        # rand_sign = torch.where((shrink_mask * (1 - negative_gt_mask) * mask).bool(), \
        #                         rand_sign * upper_bound / (upper_bound+1) / rand_part, rand_sign)
        # 把噪声加到边界框四条边
        known_bbox += rand_sign * rand_part * diff
        # 将坐标限制在合法图像范围
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        # 处理负数宽高
        input_query_bbox[input_query_bbox < 0] *= -1
        # 转换到未激活空间
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    # 生成类别 Query 特征
    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    # 正常检测 Query 不能看到去噪 Query
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    # 不同去噪组之间不能互相看到
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    # 生成去噪元信息
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
