"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

# DFINE 类的导入与注册
# 导入 PyTorch 神经网络模块库，所有网络层均继承自 nn.Module
import torch.nn as nn

# 从项目核心库导入 @register() 注册器装饰器
from ...core import register

__all__ = [
    "DFINE",
]


# 将 DFINE 类注册到系统的组件工厂中，便于按 YAML 配置文件动态实例化
@register()
class DFINE(nn.Module):
    __inject__ = [
        # 指定依赖注入的子模块名称，系统解析配置时会自动创建 backbone、encoder 和 decoder 并注入
        "backbone",
        "encoder",
        "decoder",
    ]

    # DFINE 的构造函数与前向传播
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        # 调用父类构造方法完成 PyTorch 基础节点初始化
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    # 主前向传播入口
    # x：输入的图像 Batch Tensor，形状为 [B, 3, H, W]
    def forward(self, x, targets=None):
        x = self.backbone(x) # 提取基础多尺度特征图 $C3, C4, C5$
        x = self.encoder(x) # 跨尺度特征金字塔融合，输出 $P3, P4, P5$
        x = self.decoder(x, targets) # 执行 Top-K 选框和 6 层 FDR 细粒度分布回归

        return x

    # 部署模式重参数化 convert_to_deploy
    def deploy(
        self,
    ):
        self.eval() # 锁定 Batch Normalization 均值方差并关闭 Dropout
        for m in self.modules(): # 递归遍历网络中所有子模块
            if hasattr(m, "convert_to_deploy"): # 查找拥有重参数化钩子的模块（如 ConvNormLayer_fuse）
                m.convert_to_deploy() # 将卷积权重与 BN 层参数等价融合为单个 nn.Conv2d，大幅提升 TensorRT 部署速度
        return self
