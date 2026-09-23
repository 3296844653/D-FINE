"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
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


class DySample(nn.Module):
    """Content-adaptive 2x upsampling used by HF-D-FINE CAF.

    Adapted from the official Apache-2.0 HF-DFINE implementation:
    https://github.com/HaveFun4ever/HF-DFINE
    """

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=True):
        super().__init__()
        assert style in ("lp", "pl")
        assert in_channels >= groups and in_channels % groups == 0
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0

        self.scale = int(scale)
        self.style = style
        self.groups = int(groups)
        offset_in_channels = in_channels // scale**2 if style == "pl" else in_channels
        offset_channels = 2 * groups if style == "pl" else 2 * groups * scale**2
        self.offset = nn.Conv2d(offset_in_channels, offset_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, offset_channels, 1)
            nn.init.constant_(self.scope.weight, 0.0)
            nn.init.constant_(self.scope.bias, 0.0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
            dtype=torch.float32,
        ) / self.scale
        grid_y, grid_x = torch.meshgrid(h, h, indexing="ij")
        return (
            torch.stack((grid_x, grid_y))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x, offset):
        batch_size, _, height, width = offset.shape
        offset = offset.view(batch_size, 2, -1, height, width)
        coord_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coord_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(coord_h, coord_w, indexing="ij")
        coords = torch.stack((grid_x, grid_y)).unsqueeze(0).unsqueeze(2)
        normalizer = x.new_tensor([width, height]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(
            coords.reshape(batch_size, -1, height, width), self.scale
        )
        coords = coords.view(
            batch_size, 2, -1, self.scale * height, self.scale * width
        )
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(batch_size * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.reshape(
            batch_size, -1, self.scale * height, self.scale * width
        )

    def forward(self, x):
        if self.style == "pl":
            shuffled = F.pixel_shuffle(x, self.scale)
            offset = self.offset(shuffled)
            if hasattr(self, "scope"):
                offset = offset * self.scope(shuffled).sigmoid()
            offset = F.pixel_unshuffle(offset, self.scale) * 0.5 + self.init_pos
        else:
            offset = self.offset(x)
            if hasattr(self, "scope"):
                offset = offset * self.scope(x).sigmoid() * 0.5
            else:
                offset = offset * 0.25
            offset = offset + self.init_pos
        return self._sample(x, offset)


class ChannelAdaptiveFusion(nn.Module):
    """HF-D-FINE upsampling CAF for two adjacent pyramid levels."""

    def __init__(
        self,
        high_channels,
        low_channels,
        output_channels,
        gamma=2,
        bias=1,
        groups=4,
        dyscope=True,
        act="silu",
    ):
        super().__init__()
        self.high_channels = int(high_channels)
        self.low_channels = int(low_channels)
        self.avg_high = nn.AdaptiveAvgPool2d(1)
        self.avg_low = nn.AdaptiveAvgPool2d(1)

        def kernel_size(channels):
            size = int(abs((math.log(channels, 2) + bias) / gamma))
            return size if size % 2 else size + 1

        high_kernel = kernel_size(high_channels)
        low_kernel = kernel_size(low_channels)
        joint_kernel = kernel_size(high_channels + low_channels)
        self.conv_high = nn.Conv1d(
            1, 1, high_kernel, padding=(high_kernel - 1) // 2, bias=False
        )
        self.conv_low = nn.Conv1d(
            1, 1, low_kernel, padding=(low_kernel - 1) // 2, bias=False
        )
        self.conv_joint = nn.Conv1d(
            1, 1, joint_kernel, padding=(joint_kernel - 1) // 2, bias=False
        )
        self.upsample = DySample(low_channels, groups=groups, dyscope=dyscope)
        self.align = (
            ConvNormLayer_fuse(low_channels, high_channels, 1, 1, act=act)
            if low_channels != high_channels
            else nn.Identity()
        )
        self.output = (
            ConvNormLayer_fuse(high_channels, output_channels, 1, 1, act=act)
            if high_channels != output_channels
            else nn.Identity()
        )

    @staticmethod
    def _channel_descriptor(pool, conv, feature):
        descriptor = pool(feature).squeeze(-1).transpose(-1, -2)
        return conv(descriptor).transpose(-1, -2).unsqueeze(-1)

    def forward(self, features):
        high_feature, low_feature = features
        high_desc = self._channel_descriptor(
            self.avg_high, self.conv_high, high_feature
        )
        low_desc = self._channel_descriptor(self.avg_low, self.conv_low, low_feature)
        joint = torch.cat((high_desc, low_desc), dim=1)
        joint = self.conv_joint(joint.squeeze(-1).transpose(-1, -2))
        attention = joint.transpose(-1, -2).unsqueeze(-1).sigmoid()
        high_weight, low_weight = torch.split(
            attention, [self.high_channels, self.low_channels], dim=1
        )
        high_out = high_feature * high_weight
        low_out = self.align(self.upsample(low_feature * low_weight))
        if high_out.shape[-2:] != low_out.shape[-2:]:
            raise ValueError("CAF inputs must have an exact 2x spatial relationship")
        return self.output(high_out + low_out)


class ChannelAdaptiveFusionDown(nn.Module):
    """HF-D-FINE downsampling CAF for the bottom-up PAN path."""

    def __init__(
        self,
        low_channels,
        high_channels,
        output_channels,
        gamma=2,
        bias=1,
        act="silu",
    ):
        super().__init__()
        self.low_channels = int(low_channels)
        self.high_channels = int(high_channels)
        self.avg_low = nn.AdaptiveAvgPool2d(1)
        self.avg_high = nn.AdaptiveAvgPool2d(1)

        def kernel_size(channels):
            size = int(abs((math.log(channels, 2) + bias) / gamma))
            return size if size % 2 else size + 1

        low_kernel = kernel_size(low_channels)
        high_kernel = kernel_size(high_channels)
        joint_kernel = kernel_size(low_channels + high_channels)
        self.conv_low = nn.Conv1d(
            1, 1, low_kernel, padding=(low_kernel - 1) // 2, bias=False
        )
        self.conv_high = nn.Conv1d(
            1, 1, high_kernel, padding=(high_kernel - 1) // 2, bias=False
        )
        self.conv_joint = nn.Conv1d(
            1, 1, joint_kernel, padding=(joint_kernel - 1) // 2, bias=False
        )
        self.downsample = nn.Conv2d(high_channels, low_channels, 3, 2, 1)
        self.output = (
            ConvNormLayer_fuse(low_channels, output_channels, 1, 1, act=act)
            if low_channels != output_channels
            else nn.Identity()
        )

    @staticmethod
    def _channel_descriptor(pool, conv, feature):
        descriptor = pool(feature).squeeze(-1).transpose(-1, -2)
        return conv(descriptor).transpose(-1, -2).unsqueeze(-1)

    def forward(self, features):
        low_feature, high_feature = features
        low_desc = self._channel_descriptor(self.avg_low, self.conv_low, low_feature)
        high_desc = self._channel_descriptor(
            self.avg_high, self.conv_high, high_feature
        )
        joint = torch.cat((low_desc, high_desc), dim=1)
        joint = self.conv_joint(joint.squeeze(-1).transpose(-1, -2))
        attention = joint.transpose(-1, -2).unsqueeze(-1).sigmoid()
        low_weight, high_weight = torch.split(
            attention, [self.low_channels, self.high_channels], dim=1
        )
        low_out = low_feature * low_weight
        high_out = self.downsample(high_feature * high_weight)
        if low_out.shape[-2:] != high_out.shape[-2:]:
            raise ValueError("CAF_Down inputs must have an exact 2x spatial relationship")
        return self.output(low_out + high_out)


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


class TripletZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, dim=1, keepdim=True).values, torch.mean(x, dim=1, keepdim=True)),
            dim=1,
        )


class TripletAttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1
        self.compress = TripletZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.conv(self.compress(x)))


class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7, no_spatial=False):
        super().__init__()
        self.cw = TripletAttentionGate(kernel_size)
        self.hc = TripletAttentionGate(kernel_size)
        self.no_spatial = bool(no_spatial)
        if not self.no_spatial:
            self.hw = TripletAttentionGate(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
        if self.no_spatial:
            return 0.5 * (x_out1 + x_out2)
        return (self.hw(x) + x_out1 + x_out2) / 3.0


class SimAMFeatureEnhancer(nn.Module):
    """SimAM and local-global adaptive residual SimAM for multi-scale features.

    ``standard`` preserves the original parameter-free implementation.
    ``adaptive_residual`` learns an independent residual strength for every
    feature level. ``local_global_adaptive_residual`` additionally introduces
    a local energy correction. Both modes initialize exactly as ``standard``
    SimAM.
    """

    def __init__(
        self,
        e_lambda=1e-4,
        num_levels=3,
        mode="standard",
        local_kernel_size=3,
        residual_init=1.0,
    ):
        super().__init__()
        assert mode in (
            "standard",
            "adaptive_residual",
            "local_global_adaptive_residual",
        ), (
            "simam_mode must be 'standard', 'adaptive_residual', or "
            "'local_global_adaptive_residual'"
        )
        assert int(num_levels) > 0, "simam num_levels must be positive"
        assert int(local_kernel_size) > 0 and int(local_kernel_size) % 2 == 1, (
            "simam_local_kernel_size must be a positive odd number"
        )
        assert 0.0 < float(residual_init) < 2.0, (
            "simam_residual_init must be between 0 and 2"
        )

        self.e_lambda = float(e_lambda)
        self.num_levels = int(num_levels)
        self.mode = mode
        self.local_kernel_size = int(local_kernel_size)
        self.activation = nn.Sigmoid()

        if self.mode == "local_global_adaptive_residual":

            self.local_mix = nn.Parameter(torch.zeros(self.num_levels))
        else:
            self.local_mix = None

        if self.mode in ("adaptive_residual", "local_global_adaptive_residual"):
            residual_ratio = float(residual_init) / 2.0
            residual_logit = torch.logit(torch.tensor(residual_ratio)).item()
            self.residual_scale_logits = nn.Parameter(
                torch.full((self.num_levels,), residual_logit)
            )
        else:
            self.residual_scale_logits = None

    def _global_attention(self, feat):
        height, width = feat.shape[-2:]
        num_pixels = max(height * width - 1, 1)
        residual = feat - feat.mean(dim=(2, 3), keepdim=True)
        residual_square = residual.pow(2)
        variance = residual_square.sum(dim=(2, 3), keepdim=True) / num_pixels
        energy = residual_square / (4.0 * (variance + self.e_lambda)) + 0.5
        return self.activation(energy)

    def _local_attention(self, feat):
        padding = self.local_kernel_size // 2
        local_mean = F.avg_pool2d(
            feat,
            self.local_kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        local_residual_square = (feat - local_mean).pow(2)
        local_variance = F.avg_pool2d(
            local_residual_square,
            self.local_kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        local_energy = local_residual_square / (
            4.0 * (local_variance + self.e_lambda)
        ) + 0.5
        return self.activation(local_energy)

    def forward(self, feats):
        assert len(feats) == self.num_levels, (
            f"SimAM expected {self.num_levels} feature levels, got {len(feats)}"
        )
        enhanced_feats = []
        for level, feat in enumerate(feats):
            global_attention = self._global_attention(feat)
            if self.mode == "standard":
                enhanced_feats.append(feat * global_attention)
                continue

            if self.mode == "local_global_adaptive_residual":
                local_attention = self._local_attention(feat)
                local_mix = torch.tanh(self.local_mix[level]).to(dtype=feat.dtype)
                attention = global_attention + local_mix * (
                    local_attention - global_attention
                )
            else:
                attention = global_attention

            residual_scale = 2.0 * torch.sigmoid(
                self.residual_scale_logits[level]
            ).to(dtype=feat.dtype)
            simam_feat = feat * attention
            enhanced_feats.append(feat + residual_scale * (simam_feat - feat))

        return enhanced_feats


class MBDEDepthwiseBranch(nn.Module):
    """Depthwise-pointwise detail branch used only by MBDE-P3."""

    def __init__(self, hidden_dim, dilation=1, act="silu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
        )

    def forward(self, feature):
        return self.block(feature)


class MultiScaleBehaviorDetailEnhancer(nn.Module):
    """MBDE-P3: dynamically enhance local behavior details on the P3 feature.

    The identity branch preserves the original representation, while standard
    and dilated depthwise branches capture hand/object details at two receptive
    fields. A zero-start residual scale makes the enabled model identical to
    the baseline at initialization.
    """

    def __init__(self, hidden_dim=256, reduction=4, init_scale=0.0, act="silu"):
        super().__init__()
        mid_dim = max(hidden_dim // reduction, 16)
        self.branches = nn.ModuleList(
            [
                nn.Identity(),
                MBDEDepthwiseBranch(hidden_dim, dilation=1, act=act),
                MBDEDepthwiseBranch(hidden_dim, dilation=2, act=act),
            ]
        )
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, mid_dim, 1),
            get_activation(act),
            nn.Conv2d(mid_dim, len(self.branches), 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feature):
        weights = F.softmax(self.branch_gate(feature), dim=1)
        detail = torch.zeros_like(feature)
        for branch_index, branch in enumerate(self.branches):
            detail = detail + branch(feature) * weights[:, branch_index : branch_index + 1]
        detail = self.fusion(detail)
        scale = torch.tanh(self.residual_scale).to(dtype=feature.dtype)
        return feature + scale * detail


class BMEFSDepthwiseBranch(nn.Module):
    """Lightweight depthwise branch used by BMEFS for expanded local context."""

    def __init__(self, hidden_dim, kernel_size=3, dilation=1, act="silu"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
        )

    def forward(self, x):
        return self.block(x)


class BMEFSLevelExpansion(nn.Module):
    """BMEFS level expansion with dynamic multi-receptive-field multiplexing."""

    def __init__(self, hidden_dim, act="silu", reduction=4):
        super().__init__()
        mid_dim = max(hidden_dim // reduction, 16)
        self.branches = nn.ModuleList(
            [
                nn.Identity(),
                BMEFSDepthwiseBranch(hidden_dim, kernel_size=3, dilation=1, act=act),
                BMEFSDepthwiseBranch(hidden_dim, kernel_size=3, dilation=2, act=act),
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, mid_dim, 1),
            get_activation(act),
            nn.Conv2d(mid_dim, len(self.branches), 1),
        )

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=1)
        branch_feats = [branch(x) for branch in self.branches]
        out = 0
        for branch_idx, branch_feat in enumerate(branch_feats):
            out = out + branch_feat * weights[:, branch_idx : branch_idx + 1]
        return out


class BehaviorMultiplexedExpandedFeatureSet(nn.Module):
    """BMEFS: dynamic expanded multi-scale feature set for behavior detection."""

    def __init__(
        self,
        hidden_dim=256,
        num_levels=3,
        act="silu",
        reduction=4,
        init_scale=0.01,
        cross_scale=True,
    ):
        super().__init__()
        self.cross_scale = cross_scale
        self.expansions = nn.ModuleList(
            [
                BMEFSLevelExpansion(hidden_dim, act=act, reduction=reduction)
                for _ in range(num_levels)
            ]
        )
        self.cross_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_levels)
            ]
        )
        self.gamma = nn.Parameter(torch.full((num_levels,), float(init_scale)))

    def forward(self, feats):
        expanded_feats = [expand(feat) for expand, feat in zip(self.expansions, feats)]
        out_feats = []

        for level, feat in enumerate(feats):
            context = expanded_feats[level]
            count = 1
            if self.cross_scale and level > 0:
                context = context + F.interpolate(
                    expanded_feats[level - 1], size=feat.shape[-2:], mode="nearest"
                )
                count += 1
            if self.cross_scale and level + 1 < len(feats):
                context = context + F.interpolate(
                    expanded_feats[level + 1], size=feat.shape[-2:], mode="nearest"
                )
                count += 1
            context = context / count

            gate = self.cross_gates[level](torch.cat([feat, context], dim=1))
            scale = self.gamma[level].to(dtype=feat.dtype)
            out_feats.append(feat + scale * gate * (context - feat))

        return out_feats


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
        use_spatial_prior_attn=False,
        spatial_prior_beta_min=0.75,
        spatial_prior_beta_max=1.0,
        spatial_prior_query_chunk_size=128,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.use_spatial_prior_attn = bool(use_spatial_prior_attn)
        self.spatial_prior_query_chunk_size = int(spatial_prior_query_chunk_size)
        if not 0.0 < spatial_prior_beta_min < spatial_prior_beta_max <= 1.0:
            raise ValueError(
                "spatial prior beta range must satisfy 0 < beta_min < beta_max <= 1"
            )
        if self.spatial_prior_query_chunk_size <= 0:
            raise ValueError("spatial_prior_query_chunk_size must be positive")

        # DFormerv2 assigns a different fixed decay rate to every attention
        # head. The upper endpoint is excluded, matching the paper's default
        # linear range [0.75, 1.0). This non-persistent buffer adds no model
        # parameters or checkpoint keys.
        head_indices = torch.arange(nhead, dtype=torch.float32)
        spatial_prior_betas = spatial_prior_beta_min + (
            spatial_prior_beta_max - spatial_prior_beta_min
        ) * head_indices / nhead
        self.register_buffer(
            "spatial_prior_betas", spatial_prior_betas, persistent=False
        )

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

    def _spatial_prior_attention(self, query, key, value, spatial_shape, attn_mask=None):
        """Apply paper Eq. (4): (softmax(QK^T) * beta^S) V.

        S is the pairwise Manhattan distance between 2-D feature tokens. Query
        rows are processed in chunks to avoid constructing one additional
        batch-sized HW-by-HW prior tensor. Existing MultiheadAttention
        projection parameters are reused unchanged.
        """
        if spatial_shape is None:
            raise ValueError("spatial_shape is required for spatial-prior attention")
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        batch_size, sequence_length, embed_dim = query.shape
        if height * width != sequence_length:
            raise ValueError(
                "spatial_shape does not match the flattened token sequence: "
                f"{height}x{width} != {sequence_length}"
            )
        if not self.self_attn._qkv_same_embed_dim:
            raise ValueError("spatial-prior attention requires equal Q/K/V dimensions")
        if self.self_attn.bias_k is not None or self.self_attn.bias_v is not None:
            raise ValueError("spatial-prior attention does not support bias_k/bias_v")
        if self.self_attn.add_zero_attn:
            raise ValueError("spatial-prior attention does not support add_zero_attn")

        q_weight, k_weight, v_weight = self.self_attn.in_proj_weight.chunk(3, dim=0)
        if self.self_attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.self_attn.in_proj_bias.chunk(3, dim=0)
        query = F.linear(query, q_weight, q_bias)
        key = F.linear(key, k_weight, k_bias)
        value = F.linear(value, v_weight, v_bias)

        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        query = query.reshape(batch_size, sequence_length, num_heads, head_dim)
        query = query.transpose(1, 2) * head_dim**-0.5
        key = key.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

        row_indices = torch.arange(height, device=query.device)
        col_indices = torch.arange(width, device=query.device)
        grid_y, grid_x = torch.meshgrid(row_indices, col_indices, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x), dim=-1).reshape(sequence_length, 2)
        log_beta = self.spatial_prior_betas.log().to(device=query.device)

        output_chunks = []
        chunk_size = min(self.spatial_prior_query_chunk_size, sequence_length)
        for start in range(0, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            logits = torch.matmul(query[:, :, start:end], key.transpose(-2, -1))

            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    mask = attn_mask[start:end].unsqueeze(0).unsqueeze(0)
                elif attn_mask.ndim == 3 and attn_mask.shape[0] == batch_size * num_heads:
                    mask = attn_mask.reshape(
                        batch_size, num_heads, sequence_length, sequence_length
                    )[:, :, start:end]
                else:
                    raise ValueError("unsupported attention-mask shape")
                if mask.dtype == torch.bool:
                    logits = logits.masked_fill(
                        mask.to(device=logits.device), float("-inf")
                    )
                else:
                    logits = logits + mask.to(dtype=logits.dtype, device=logits.device)

            attention = logits.softmax(dim=-1)
            distance = (
                coordinates[start:end, None] - coordinates[None, :]
            ).abs().sum(dim=-1)
            spatial_decay = torch.exp(
                log_beta[:, None, None] * distance.to(dtype=log_beta.dtype)[None]
            ).to(dtype=attention.dtype)

            # Deliberately do not renormalize after multiplication: this is the
            # exact ordering written in DFormerv2 Eq. (4).
            attention = attention * spatial_decay.unsqueeze(0)
            attention = F.dropout(
                attention,
                p=self.self_attn.dropout,
                training=self.training,
            )
            output_chunks.append(torch.matmul(attention, value))

        output = torch.cat(output_chunks, dim=2)
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, embed_dim)
        return self.self_attn.out_proj(output)

    def forward(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        spatial_shape=None,
    ) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if self.use_spatial_prior_attn:
            src = self._spatial_prior_attention(
                q,
                k,
                src,
                spatial_shape=spatial_shape,
                attn_mask=src_mask,
            )
        else:
            # Preserve the original D-FINE path exactly when the switch is off.
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

    def forward(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        spatial_shape=None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                pos_embed=pos_embed,
                spatial_shape=spatial_shape,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class FixedQuerySelfAttention(nn.Module):
    """Fixed-Query Self-Attention (FQSA) from AQF-Net.

    The paper specifies fixed-resolution adaptive pooling, a lightweight local
    query branch, a pyramidal multi-scale key/value branch, multi-head scaled
    dot-product attention, and bilinear restoration. It does not disclose the
    exact kernels used inside the two branches; the depth-wise 3x3 and dilated
    3x3 operations below are deliberately exposed as implementation choices.
    """

    def __init__(
        self,
        channels,
        num_heads=8,
        query_size=16,
        pyramid_dilation=2,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("FQSA channels must be divisible by num_heads")
        if query_size <= 0:
            raise ValueError("FQSA query_size must be positive")
        if pyramid_dilation <= 0:
            raise ValueError("FQSA pyramid_dilation must be positive")

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.query_size = int(query_size)

        self.pool = nn.AdaptiveAvgPool2d((self.query_size, self.query_size))
        self.query_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.query_local = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
        )

        self.kv_local = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
        )
        self.kv_context = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=pyramid_dilation,
            dilation=pyramid_dilation,
            groups=self.channels,
        )
        self.kv_proj = nn.Conv2d(self.channels, self.channels * 2, kernel_size=1)
        self.output_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)

    def _reshape_heads(self, tensor):
        batch_size, _, height, width = tensor.shape
        return tensor.reshape(
            batch_size, self.num_heads, self.head_dim, height * width
        ).transpose(-2, -1)

    def forward(self, x):
        _, _, height, width = x.shape
        pooled = self.pool(x)

        query = self.query_proj(pooled)
        query = query + self.query_local(query)

        pyramid = pooled + self.kv_local(pooled) + self.kv_context(pooled)
        key, value = self.kv_proj(pyramid).chunk(2, dim=1)

        query = self._reshape_heads(query)
        key = self._reshape_heads(key)
        value = self._reshape_heads(value)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim**-0.5
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(-2, -1).reshape(
            x.shape[0], self.channels, self.query_size, self.query_size
        )
        output = self.output_proj(output)
        return F.interpolate(
            output,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )


class BidirectionalSelectiveAggregation(nn.Module):
    """Channel-wise and spatial-wise branch selection used by AQF-Net LREA."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.channel_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.channel_depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.channel_expand = nn.ModuleList(
            [nn.Conv2d(hidden_channels, channels, kernel_size=1) for _ in range(3)]
        )
        self.spatial_map = nn.Conv2d(2, 3, kernel_size=7, padding=3)

    def forward(self, branch_features):
        if len(branch_features) != 3:
            raise ValueError("BSA expects exactly three branch features")
        shared = branch_features[0] + branch_features[1] + branch_features[2]

        channel_context = F.adaptive_avg_pool2d(shared, 1)
        channel_context = self.channel_reduce(channel_context)
        channel_context = self.channel_depthwise(channel_context)
        channel_weights = [
            torch.sigmoid(expand(channel_context)) for expand in self.channel_expand
        ]

        spatial_context = torch.cat(
            [shared.mean(dim=1, keepdim=True), shared.amax(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_weights = torch.sigmoid(self.spatial_map(spatial_context)).chunk(3, dim=1)

        return sum(
            feature * channel_weight * spatial_weight
            for feature, channel_weight, spatial_weight in zip(
                branch_features, channel_weights, spatial_weights
            )
        )


class SparseDecomposedBroadConvolution(nn.Module):
    """SDBConv: local 5x5 DWConv plus horizontal/vertical dilated strips."""

    def __init__(self, channels, strip_kernel_size=11, strip_dilation=2, reduction=4):
        super().__init__()
        if strip_kernel_size % 2 != 1:
            raise ValueError("LREA strip_kernel_size must be odd")
        if strip_dilation <= 0:
            raise ValueError("LREA strip_dilation must be positive")

        strip_padding = strip_dilation * (strip_kernel_size - 1) // 2
        self.local = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_kernel_size),
            padding=(0, strip_padding),
            dilation=(1, strip_dilation),
            groups=channels,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_kernel_size, 1),
            padding=(strip_padding, 0),
            dilation=(strip_dilation, 1),
            groups=channels,
        )
        self.selection = BidirectionalSelectiveAggregation(channels, reduction)
        self.modulation = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        local_feature = self.local(x)
        horizontal_feature = self.horizontal(local_feature)
        vertical_feature = self.vertical(local_feature)
        fused = self.selection(
            [local_feature, horizontal_feature, vertical_feature]
        )
        return x * self.modulation(fused)


class LargeReceptiveFieldAttention(nn.Module):
    def __init__(self, channels, strip_kernel_size=11, strip_dilation=2, reduction=4):
        super().__init__()
        self.input_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sdbconv = SparseDecomposedBroadConvolution(
            channels,
            strip_kernel_size=strip_kernel_size,
            strip_dilation=strip_dilation,
            reduction=reduction,
        )
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return self.output_proj(self.sdbconv(F.gelu(self.input_proj(x))))


class ConvolutionalFeedForwardNetwork(nn.Module):
    def __init__(self, channels, expansion=2.0):
        super().__init__()
        hidden_channels = max(int(round(channels * expansion)), channels)
        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.compress = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.expand(x)
        x = F.gelu(self.depthwise(x))
        return self.compress(x)


class LargeReceptiveFieldEnhancement(nn.Module):
    """LREA attention-feedforward block from AQF-Net."""

    def __init__(
        self,
        channels,
        strip_kernel_size=11,
        strip_dilation=2,
        bsa_reduction=4,
        cffn_expansion=2.0,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = LargeReceptiveFieldAttention(
            channels,
            strip_kernel_size=strip_kernel_size,
            strip_dilation=strip_dilation,
            reduction=bsa_reduction,
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.feed_forward = ConvolutionalFeedForwardNetwork(
            channels, expansion=cffn_expansion
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


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
        use_hf_gate=False,
        hf_gate_init=-3.0,
        hf_gate_level=0,
        hf_return_indices=None,
        use_simam=False,
        simam_e_lambda=1e-4,
        simam_mode="standard",
        simam_local_kernel_size=3,
        simam_residual_init=1.0,
        use_bmefs=False,
        bmefs_init=0.01,
        bmefs_reduction=4,
        bmefs_cross_scale=True,
        use_mbde_p3=False,
        mbde_p3_init=0.0,
        mbde_p3_reduction=4,
        use_triplet_attention=False,
        triplet_attention_kernel_size=7,
        triplet_attention_levels=None,
        triplet_attention_no_spatial=False,
        use_caf=False,
        caf_gamma=2,
        caf_bias=1,
        caf_groups=4,
        caf_dyscope=True,
        use_sfif=False,
        sfif_num_heads=8,
        use_fqsa=False,
        fqsa_query_size=16,
        fqsa_num_heads=8,
        fqsa_pyramid_dilation=2,
        use_lrea=False,
        lrea_strip_kernel_size=11,
        lrea_strip_dilation=2,
        lrea_bsa_reduction=4,
        lrea_cffn_expansion=2.0,
        use_spatial_prior_attn=False,
        spatial_prior_beta_min=0.75,
        spatial_prior_beta_max=1.0,
        spatial_prior_query_chunk_size=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx

        if self.use_encoder_idx:
            assert max(self.use_encoder_idx) < len(in_channels), "use_encoder_idx exceeds feature levels"
        if hf_return_indices is not None:
            assert len(hf_return_indices) > 0, "hf_return_indices cannot be empty"
            assert max(hf_return_indices) < len(in_channels), "hf_return_indices exceeds feature levels"
        if use_hf_gate:
            assert 0 <= hf_gate_level < len(in_channels) - 1, "hf_gate_level must target a fused lower level"

        self.use_hf_gate = use_hf_gate
        self.hf_gate_level = hf_gate_level
        self.hf_return_indices = hf_return_indices
        self.use_bmefs = use_bmefs
        self.use_simam = use_simam
        self.use_mbde_p3 = use_mbde_p3
        self.use_triplet_attention = bool(use_triplet_attention)
        self.use_caf = bool(use_caf)
        self.use_sfif = bool(use_sfif)
        self.use_fqsa = bool(use_fqsa)
        self.use_lrea = bool(use_lrea)
        self.use_spatial_prior_attn = bool(use_spatial_prior_attn)
        if self.use_sfif and self.use_fqsa:
            raise ValueError("SFIF and FQSA are alternative encoder replacements")
        if self.use_spatial_prior_attn and (self.use_sfif or self.use_fqsa):
            raise ValueError(
                "spatial-prior attention requires the original AIFI encoder; "
                "disable SFIF and FQSA"
            )
        if self.use_sfif and num_encoder_layers != 1:
            raise ValueError("SFIF replaces the single AIFI layer; set num_encoder_layers to 1")
        if self.use_sfif and hidden_dim % sfif_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by sfif_num_heads")
        if self.use_fqsa and num_encoder_layers != 1:
            raise ValueError("FQSA replaces the single AIFI layer; set num_encoder_layers to 1")
        if self.use_fqsa and hidden_dim % fqsa_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by fqsa_num_heads")
        if triplet_attention_levels is None:
            triplet_attention_levels = list(range(len(in_channels)))
        self.triplet_attention_levels = tuple(int(level) for level in triplet_attention_levels)
        assert len(set(self.triplet_attention_levels)) == len(self.triplet_attention_levels)
        assert all(0 <= level < len(in_channels) for level in self.triplet_attention_levels)
        if self.use_hf_gate:
            self.hf_gate = nn.Parameter(torch.tensor(float(hf_gate_init)))
        else:
            self.hf_gate = None

        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        if self.hf_return_indices is not None:
            self.out_channels = [hidden_dim for _ in self.hf_return_indices]
            self.out_strides = [feat_strides[i] for i in self.hf_return_indices]
        else:
            self.out_channels = [hidden_dim for _ in range(len(in_channels))]
            self.out_strides = feat_strides

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

        # SFIF and FQSA are alternative single-variable replacements for AIFI,
        # rather than extra blocks stacked on top of the baseline transformer.
        if self.use_fqsa:
            self.encoder = nn.ModuleList(
                [
                    FixedQuerySelfAttention(
                        hidden_dim,
                        num_heads=fqsa_num_heads,
                        query_size=fqsa_query_size,
                        pyramid_dilation=fqsa_pyramid_dilation,
                    )
                    for _ in range(len(use_encoder_idx))
                ]
            )
        elif self.use_sfif:
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
                use_spatial_prior_attn=self.use_spatial_prior_attn,
                spatial_prior_beta_min=spatial_prior_beta_min,
                spatial_prior_beta_max=spatial_prior_beta_max,
                spatial_prior_query_chunk_size=spatial_prior_query_chunk_size,
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
        self.caf_ups = nn.ModuleList()
        self.fpn_lrea = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            if self.use_caf:
                self.caf_ups.append(
                    ChannelAdaptiveFusion(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        gamma=caf_gamma,
                        bias=caf_bias,
                        groups=caf_groups,
                        dyscope=caf_dyscope,
                        act=act,
                    )
                )
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim if self.use_caf else hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
                # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )
            if self.use_lrea:
                self.fpn_lrea.append(
                    LargeReceptiveFieldEnhancement(
                        hidden_dim,
                        strip_kernel_size=lrea_strip_kernel_size,
                        strip_dilation=lrea_strip_dilation,
                        bsa_reduction=lrea_bsa_reduction,
                        cffn_expansion=lrea_cffn_expansion,
                    )
                )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.caf_downs = nn.ModuleList()
        self.pan_lrea = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            if self.use_caf:
                self.caf_downs.append(
                    ChannelAdaptiveFusionDown(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        gamma=caf_gamma,
                        bias=caf_bias,
                        act=act,
                    )
                )
            else:
                self.downsample_convs.append(
                    nn.Sequential(
                        SCDown(hidden_dim, hidden_dim, 3, 2),
                    )
                )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim if self.use_caf else hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
                # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )
            if self.use_lrea:
                self.pan_lrea.append(
                    LargeReceptiveFieldEnhancement(
                        hidden_dim,
                        strip_kernel_size=lrea_strip_kernel_size,
                        strip_dilation=lrea_strip_dilation,
                        bsa_reduction=lrea_bsa_reduction,
                        cffn_expansion=lrea_cffn_expansion,
                    )
                )


        if self.use_simam:
            self.simam = SimAMFeatureEnhancer(
                e_lambda=simam_e_lambda,
                num_levels=len(in_channels),
                mode=simam_mode,
                local_kernel_size=simam_local_kernel_size,
                residual_init=simam_residual_init,
            )
        else:
            self.simam = None


        if self.use_bmefs:
            self.bmefs = BehaviorMultiplexedExpandedFeatureSet(
                hidden_dim=hidden_dim,
                num_levels=len(in_channels),
                act=act,
                reduction=bmefs_reduction,
                init_scale=bmefs_init,
                cross_scale=bmefs_cross_scale,
            )
        else:
            self.bmefs = None


        if self.use_mbde_p3:
            self.mbde_p3 = MultiScaleBehaviorDetailEnhancer(
                hidden_dim=hidden_dim,
                reduction=mbde_p3_reduction,
                init_scale=mbde_p3_init,
                act=act,
            )
        else:
            self.mbde_p3 = None

        if self.use_triplet_attention:
            self.triplet_attention = nn.ModuleDict(
                {
                    str(level): TripletAttention(
                        kernel_size=triplet_attention_kernel_size,
                        no_spatial=triplet_attention_no_spatial,
                    )
                    for level in self.triplet_attention_levels
                }
            )
        else:
            self.triplet_attention = None

        self._reset_parameters()

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
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                if self.use_sfif or self.use_fqsa:
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

                memory: torch.Tensor = self.encoder[i](
                    src_flatten,
                    pos_embed=pos_embed,
                    spatial_shape=(h, w),
                )
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
            fusion_idx = len(self.in_channels) - 1 - idx
            if self.use_caf:
                fused_input = self.caf_ups[fusion_idx]([feat_low, feat_heigh])
                fused_feat = self.fpn_blocks[fusion_idx](fused_input)
                upsample_feat = None
            else:
                upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
                fused_feat = self.fpn_blocks[fusion_idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
            if self.use_lrea:
                fused_feat = self.fpn_lrea[fusion_idx](fused_feat)
            if self.use_hf_gate and (idx - 1) == self.hf_gate_level:
                if self.use_caf:
                    raise ValueError("use_hf_gate and use_caf are mutually exclusive")
                gate = torch.sigmoid(self.hf_gate).to(dtype=fused_feat.dtype)
                inner_out = upsample_feat + gate * (fused_feat - upsample_feat)
            else:
                inner_out = fused_feat
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            if self.use_caf:
                fused_input = self.caf_downs[idx]([feat_height, feat_low])
                out = self.pan_blocks[idx](fused_input)
            else:
                downsample_feat = self.downsample_convs[idx](feat_low)
                out = self.pan_blocks[idx](
                    torch.concat([downsample_feat, feat_height], dim=1)
                )
            if self.use_lrea:
                out = self.pan_lrea[idx](out)
            outs.append(out)

        if self.use_triplet_attention:
            outs = [
                self.triplet_attention[str(level)](feat)
                if level in self.triplet_attention_levels
                else feat
                for level, feat in enumerate(outs)
            ]

        if self.use_mbde_p3:
            outs[0] = self.mbde_p3(outs[0])

        if self.use_simam:
            outs = self.simam(outs)

        if self.use_bmefs:
            outs = self.bmefs(outs)

        if self.hf_return_indices is not None:
            outs = [outs[i] for i in self.hf_return_indices]

        return outs
