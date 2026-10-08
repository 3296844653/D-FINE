"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .medium_finegrained import MediumScaleFineGrainedEnhancement
from .utils import get_activation

__all__ = ["HybridEncoder"]


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ELAN(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=2, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        # y = [self.cv1(x)]
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden_channels, hidden_channels, act=get_activation(act))
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class P3RFAConvResidual(nn.Module):
    """Official group-convolution RFAConv mechanism inside a P3 bottleneck.

    Source: github.com/Liuchen1997/RFAConv/blob/main/model.py (first RFAConv).
    Adaptations: 1x1 channel reduction/restoration and a fixed residual scale.
    Native reshape/permute replaces einops with identical spatial ordering.
    This is not the older RFAInspiredP3Refiner or the full EduYOLO architecture.
    """

    def __init__(self, channels, mid_channels=32, alpha=0.1):
        super().__init__()
        if mid_channels <= 0 or not 0 <= alpha <= 1:
            raise ValueError("P3 RFAConv requires positive width and alpha in [0, 1]")
        self.alpha = float(alpha)  # Fixed coefficient, not alpha_init.
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
        )
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(mid_channels, mid_channels * 9, 1, groups=mid_channels, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 9, 3, padding=1,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels * 9), nn.ReLU(),
        )
        self.aggregate = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=3, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
        )
        self.restore = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        reduced = self.reduce(x)
        b, c, h, w = reduced.shape
        weights = self.get_weight(reduced).reshape(b, c, 9, h, w).softmax(dim=2)
        features = self.generate_feature(reduced).reshape(b, c, 9, h, w)
        # k-index = row*3+column, matching official einops rearrange.
        expanded = (features * weights).reshape(b, c, 3, 3, h, w)
        expanded = expanded.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * 3, w * 3)
        return x + self.alpha * self.restore(self.aggregate(expanded))


class RFAInspiredP3Refiner(nn.Module):
    """Receptive-field-weighted local refinement inspired by EduYOLO's RFAConv.

    This is a bottleneck residual adaptation for D-FINE, not a reproduction of
    EduYOLO's C3RFA block. Each P3 position learns nine weights over its 3x3
    receptive field; no YOLO neck, P2 head, or regression loss is transplanted.
    The zero-initialized output projection makes the initial residual exactly 0.
    """

    def __init__(self, channels: int, mid_channels: int = 32):
        super().__init__()
        if mid_channels <= 0:
            raise ValueError("rfa_mid_channels must be positive")
        self.reduce = ConvNormLayer(channels, mid_channels, 1, 1, act="silu")
        self.offset_weights = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, mid_channels * 9, 1, groups=mid_channels, bias=False),
        )
        self.local_features = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 9, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels * 9),
            nn.SiLU(),
        )
        self.aggregate = ConvNormLayer(mid_channels, mid_channels, 3, 3, padding=0, act="silu")
        self.expand = nn.Conv2d(mid_channels, channels, 1, bias=True)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        batch, channels, height, width = reduced.shape
        weights = self.offset_weights(reduced).reshape(batch, channels, 9, height, width)
        weights = weights.softmax(dim=2)
        features = self.local_features(reduced).reshape(batch, channels, 9, height, width)
        # PixelShuffle places the nine weighted offsets in a 3H x 3W grid.
        weighted = (features * weights).reshape(batch, channels * 9, height, width)
        local_grid = F.pixel_shuffle(weighted, upscale_factor=3)
        return x + self.expand(self.aggregate(local_grid))


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class SpatioFrequencyInteractiveFusion(nn.Module):
    """Spatio-Frequency Interactive Fusion (SFIF) from FBDNet.

    The spatial branch performs channel-wise multi-head self-attention plus a
    depth-wise local value path. The frequency branch predicts a spatial
    weight map, transforms both tensors with a 2-D FFT, and filters the input
    by complex multiplication. Two cross gates then fuse both branches into a
    residual output.
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("SFIF channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.local_value = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.spatial_scale = nn.Parameter(
            torch.full((num_heads, 1, 1), self.head_dim**-0.5)
        )
        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.frequency_weight_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.frequency_weight_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.frequency_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.spatial_to_frequency_gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.frequency_to_spatial_gate = nn.Conv2d(channels, channels, kernel_size=1)

    def _spatial_branch(self, x):
        batch_size, _, height, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=1)
        local_value = self.local_value(value)

        query = query.reshape(
            batch_size, self.num_heads, self.head_dim, height * width
        )
        key = key.reshape(batch_size, self.num_heads, self.head_dim, height * width)
        value = value.reshape(
            batch_size, self.num_heads, self.head_dim, height * width
        )

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention * self.spatial_scale.to(dtype=attention.dtype)
        attention = attention.softmax(dim=-1)
        global_value = torch.matmul(attention, value).reshape(
            batch_size, self.channels, height, width
        )
        return self.spatial_proj(global_value + local_value)

    def _frequency_branch(self, x):
        frequency_weight = self.frequency_weight_out(
            F.gelu(self.frequency_weight_in(x))
        )

        # CUDA FFT does not support every low-precision shape under AMP. The
        # transform is therefore evaluated in fp32 and cast back afterwards.
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        fft_weight = (
            frequency_weight.float()
            if frequency_weight.dtype in (torch.float16, torch.bfloat16)
            else frequency_weight
        )
        filtered = torch.fft.fft2(fft_input, dim=(-2, -1)) * torch.fft.fft2(
            fft_weight, dim=(-2, -1)
        )
        restored = torch.fft.ifft2(filtered, dim=(-2, -1)).real.to(dtype=x.dtype)
        return self.frequency_proj(restored)

    def forward(self, x):
        spatial_feature = self._spatial_branch(x)
        frequency_feature = self._frequency_branch(x)

        frequency_gate = torch.sigmoid(
            self.spatial_to_frequency_gate(spatial_feature)
        )
        spatial_gate = torch.sigmoid(
            self.frequency_to_spatial_gate(frequency_feature)
        )
        fused = spatial_feature * spatial_gate + frequency_feature * frequency_gate
        return x + fused


@register()
class HybridEncoder(nn.Module):
    __share__ = [
        "eval_spatial_size",
    ]

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_rfa_p3=False,
        rfa_mid_channels=32,
        use_sfif=False,
        sfif_num_heads=8,
        use_p2_detail_fusion=False,
        p2_in_channels=64,
        use_encoder_highres_residual=False,
        encoder_highres_alpha_init=0.0,
        use_encoder_p3_joint_residual=False,
        encoder_p3_joint_dim=64,
        use_p3_rfaconv_residual=False,
        p3_rfaconv_mid_channels=32,
        p3_rfaconv_alpha=0.1,
        use_mffe=False,
        mffe_mid_channels=64,
        mffe_alpha_init=0.1,
        mffe_checkpoint=True,
        mffe_p3_only=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_sfif = bool(use_sfif)
        self.use_p2_detail_fusion = bool(use_p2_detail_fusion)
        self.use_encoder_highres_residual = bool(use_encoder_highres_residual)
        self.use_encoder_p3_joint_residual = bool(use_encoder_p3_joint_residual)
        if self.use_sfif and num_encoder_layers != 1:
            raise ValueError("SFIF replaces the single AIFI layer; set num_encoder_layers to 1")
        if self.use_sfif and hidden_dim % sfif_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by sfif_num_heads")
        self.use_rfa_p3 = use_rfa_p3
        self.use_p3_rfaconv_residual = bool(use_p3_rfaconv_residual)
        self.use_mffe = bool(use_mffe)
        if not isinstance(mffe_p3_only, bool):
            raise ValueError("mffe_p3_only must be a boolean")
        self.mffe_p3_only = mffe_p3_only
        if sum(
            (
                self.use_p2_detail_fusion,
                self.use_encoder_highres_residual,
                self.use_encoder_p3_joint_residual,
                self.use_rfa_p3,
                self.use_sfif,
                self.use_p3_rfaconv_residual,
                self.use_mffe,
            )
        ) > 1:
            raise ValueError(
                "P2 detail fusion, encoder high-resolution/joint residual, RFA-P3, P3-RFAConv, SFIF, and MFFE "
                "must be ablated separately"
            )
        if self.use_mffe and (len(in_channels) != 3 or list(feat_strides) != [8, 16, 32]):
            raise ValueError("MFFE requires the standard P3/P4/P5 encoder at strides 8/16/32")
        if self.use_p2_detail_fusion:
            if len(in_channels) != 3 or feat_strides[0] != 8 or p2_in_channels <= 0:
                raise ValueError("P2 detail fusion requires the standard P3-P5 encoder")
            # Downsample stride-4 P2 once and project it to the P3 width. The
            # depthwise step retains local spatial evidence at low cost. A
            # zero-initialized final BN makes the initial residual exactly zero.
            self.p2_detail_proj = nn.Sequential(
                nn.Conv2d(
                    p2_in_channels,
                    p2_in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=p2_in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(p2_in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(p2_in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            nn.init.zeros_(self.p2_detail_proj[-1].weight)
            nn.init.zeros_(self.p2_detail_proj[-1].bias)
        if self.use_encoder_highres_residual:
            if len(in_channels) != 3 or feat_strides[0] != 8:
                raise ValueError(
                    "Encoder high-resolution residual requires the standard P3-P5 encoder"
                )
            # A single learnable coefficient probes whether the original P3
            # information should bypass multi-scale FPN/PAN fusion. Zero init
            # guarantees exact baseline behavior before training.
            self.encoder_highres_alpha = nn.Parameter(
                torch.tensor(float(encoder_highres_alpha_init))
            )
        if self.use_encoder_p3_joint_residual:
            if len(in_channels) != 3 or feat_strides[0] != 8 or encoder_p3_joint_dim <= 0:
                raise ValueError("Joint P3 residual requires P3-P5 and a positive bottleneck width")
            # Jointly transform original projected P3 and final fused P3.
            # No normalization is added, so baseline running statistics stay intact.
            self.encoder_p3_joint_residual = nn.Sequential(
                nn.Conv2d(2 * hidden_dim, encoder_p3_joint_dim, 1),
                nn.SiLU(),
                nn.Conv2d(encoder_p3_joint_dim, hidden_dim, 1),
            )
            nn.init.zeros_(self.encoder_p3_joint_residual[-1].weight)
            nn.init.zeros_(self.encoder_p3_joint_residual[-1].bias)
        if use_rfa_p3:
            if len(in_channels) != 3 or feat_strides[0] != 8:
                raise ValueError("RFA P3 refinement requires the standard P3-P5 encoder")
            self.rfa_p3 = RFAInspiredP3Refiner(hidden_dim, rfa_mid_channels)
        if self.use_p3_rfaconv_residual:
            if len(in_channels) != 3 or feat_strides != [8, 16, 32]:
                raise ValueError("P3-RFAConv requires the standard P3/P4/P5 encoder")
            self.p3_rfaconv_residual = P3RFAConvResidual(
                hidden_dim, p3_rfaconv_mid_channels, p3_rfaconv_alpha
            )

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )

            self.input_proj.append(proj)

        # SFIF is an alternative single-variable replacement for AIFI, rather
        # than an extra block stacked on top of the baseline transformer.
        if self.use_sfif:
            self.encoder = nn.ModuleList(
                [
                    SpatioFrequencyInteractiveFusion(hidden_dim, sfif_num_heads)
                    for _ in range(len(use_encoder_idx))
                ]
            )
        else:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act,
            )

            self.encoder = nn.ModuleList(
                [
                    TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                    for _ in range(len(use_encoder_idx))
                ]
            )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
                # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    SCDown(hidden_dim, hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
                # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self._reset_parameters()

        if self.use_mffe:
            # Build AFTER the original modules and restore CPU RNG state so
            # enabling MFFE does not alter initialization of the baseline
            # encoder or decoder merely by consuming extra random numbers.
            with torch.random.fork_rng(devices=[]):
                self.mffe = nn.ModuleList(
                    MediumScaleFineGrainedEnhancement(
                        hidden_dim, mffe_mid_channels, mffe_alpha_init,
                        use_checkpoint=mffe_checkpoint,
                    )
                    # The default keeps the old P3/P4 parameter names and
                    # initialization. P3-only constructs NO unused P4 branch.
                    for _ in range(1 if self.mffe_p3_only else 2)
                )

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """ """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        if self.use_p2_detail_fusion:
            if len(feats) != len(self.in_channels) + 1:
                raise ValueError(
                    "P2 detail fusion requires backbone features [P2, P3, P4, P5]"
                )
            p2_feat, feats = feats[0], feats[1:]
        else:
            if len(feats) != len(self.in_channels):
                raise ValueError("HybridEncoder feature count does not match in_channels")
            p2_feat = None

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        mffe_sources = proj_feats[:len(self.mffe)] if self.use_mffe else None
        if self.use_p3_rfaconv_residual:
            # Stride-8 backbone P3, after channel alignment but BEFORE AIFI/FPN/PAN.
            # Keep P4/P5 inputs and all three output dimensions unchanged.
            proj_feats[0] = self.p3_rfaconv_residual(proj_feats[0])
        highres_residual = (
            proj_feats[0]
            if self.use_encoder_highres_residual or self.use_encoder_p3_joint_residual
            else None
        )

        if self.use_p2_detail_fusion:
            p2_detail = self.p2_detail_proj(p2_feat)
            if p2_detail.shape[-2:] != proj_feats[0].shape[-2:]:
                p2_detail = F.interpolate(
                    p2_detail, size=proj_feats[0].shape[-2:], mode="bilinear", align_corners=False
                )
            # Preserve the standard three-level encoder and decoder interface.
            proj_feats[0] = proj_feats[0] + p2_detail

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                if self.use_sfif:
                    proj_feats[enc_ind] = self.encoder[i](proj_feats[enc_ind])
                    continue
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                )

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        if self.use_mffe:
            # Semantic-guided detail restoration on FPN P3 (and optionally P4),
            # BEFORE PAN. P3-only skips direct P4 enhancement, but does not
            # isolate P4/P5 from P3 changes propagated through the usual PAN.
            # PAN then carries the enhanced detail into the coarser outputs;
            # no extra feature level or change to the decoder interface.
            for idx, enhancement in enumerate(self.mffe):
                inner_outs[idx] = enhancement(mffe_sources[idx], inner_outs[idx])

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        if self.use_rfa_p3:
            # Refine only the final stride-8 P3 output. P4/P5 and the decoder
            # architecture are unchanged, making this a single-variable test.
            outs[0] = self.rfa_p3(outs[0])

        if self.use_encoder_highres_residual:
            # Reintroduce the pre-fusion stride-8 feature only after FPN/PAN.
            # P4/P5 and the decoder interface remain untouched.
            outs[0] = (
                outs[0]
                + self.encoder_highres_alpha.to(outs[0].dtype) * highres_residual
            )

        if self.use_encoder_p3_joint_residual:
            outs[0] = outs[0] + self.encoder_p3_joint_residual(
                torch.cat((outs[0], highres_residual), dim=1)
            )

        return outs
