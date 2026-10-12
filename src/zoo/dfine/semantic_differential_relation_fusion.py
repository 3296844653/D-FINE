"""SDRF: Semantic Differential Relation Fusion, a research candidate.

P4 conditions the *relations* between P3 locations, never the value stream.
The correction is (A_semantic - A_detail) V_detail, not a same-location gate,
an appearance blend, anatomical supervision, or a GT-conditioned operator.
Borrowed ideas: differential attention (Differential Transformer, ICLR 2025)
and blocked attention with halo neighborhoods (HaloNet, CVPR 2021).
This is neither an official reproduction nor established original novelty.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SemanticDifferentialRelationFusion(nn.Module):
    MAX_RESIDUAL_SCALE = 0.5

    def __init__(self, channels=256, dim=64, heads=4, window=5, halo=1,
                 residual_init=0.1, use_checkpoint=True,
                 semantic_condition=True, differential=True):
        super().__init__()
        dimensions = (channels, dim, heads, window, halo)
        if any(type(value) is not int for value in dimensions):
            raise ValueError("SDRF dimensions must be integers, not booleans")
        if channels < 16 or dim < 16 or dim % 8 or heads < 1 or dim % heads:
            raise ValueError("SDRF requires channels>=16, dim>=16 divisible by 8 and heads")
        if not 3 <= window <= 9 or window % 2 == 0 or not 1 <= halo <= window // 2:
            raise ValueError("SDRF needs an odd window in [3,9] and halo in [1,window//2]")
        if not math.isfinite(residual_init) or not 0 < residual_init < self.MAX_RESIDUAL_SCALE:
            raise ValueError("SDRF residual_init must be in (0,0.5), avoiding double-zero initialization")
        if any(type(flag) is not bool for flag in (use_checkpoint, semantic_condition, differential)):
            raise ValueError("SDRF controls must be booleans")
        self.channels, self.dim, self.heads = channels, dim, heads
        self.window, self.halo = window, halo
        self.head_dim = dim // heads
        self.use_checkpoint = use_checkpoint
        self.semantic_condition, self.differential = semantic_condition, differential
        # Integer metadata is not averaged by ModelEMA or rounded by half().
        self.register_buffer("settings", torch.tensor([
            1, channels, dim, heads, window, halo, int(use_checkpoint),
            int(semantic_condition), int(differential)], dtype=torch.int64))

        def projection():
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(channels, dim, 1, bias=False)),
                ("norm", nn.GroupNorm(8, dim)), ("act", nn.SiLU()),
            ]))

        self.detail_projection = projection()
        self.semantic_projection = projection()
        self.detail_qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.semantic_query = nn.Conv2d(dim, dim, 1, bias=False)
        self.semantic_strength_logit = nn.Parameter(torch.zeros(heads))  # sigmoid -> 0.5
        self.raw_alpha = nn.Parameter(torch.tensor(math.atanh(residual_init / self.MAX_RESIDUAL_SCALE)))

        # Q locations use a block; K/V include a surrounding halo. No cyclic
        # wraparound, image-wide attention, or third-party CUDA extension.
        q_axis = torch.arange(window)
        k_axis = torch.arange(window + 2 * halo) - halo
        qy, qx = torch.meshgrid(q_axis, q_axis, indexing="ij")
        ky, kx = torch.meshgrid(k_axis, k_axis, indexing="ij")
        q = torch.stack((qy, qx), -1).reshape(-1, 2)
        k = torch.stack((ky, kx), -1).reshape(-1, 2)
        span = 2 * (window + halo) - 1
        relative = q[:, None] - k[None, :] + window + halo - 1
        self.register_buffer("relative_index", relative[..., 0] * span + relative[..., 1], persistent=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, span * span))
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fuse = nn.Sequential(OrderedDict([
            ("hidden", nn.Conv2d(dim, dim, 1, bias=False)), ("act", nn.SiLU()),
            ("output", nn.Conv2d(dim, channels, 1, bias=False)),
        ]))
        nn.init.zeros_(self.fuse.output.weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        settings = state_dict.get(prefix + "settings")
        if settings is not None and not torch.equal(settings.detach().cpu(), self.settings.detach().cpu()):
            error_msgs.append(f"{prefix}SDRF checkpoint/config settings disagree")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def residual_scale(self):
        return self.MAX_RESIDUAL_SCALE * self.raw_alpha.tanh()

    def _query_blocks(self, feature):
        batch, channels, height, width = feature.shape
        block = self.window
        return feature.reshape(batch, channels, height // block, block, width // block, block).permute(
            0, 2, 4, 3, 5, 1).reshape(-1, block * block, channels)

    def _key_blocks(self, feature):
        batch, channels, _, _ = feature.shape
        count = (self.window + 2 * self.halo) ** 2
        unfolded = F.unfold(feature, kernel_size=self.window + 2 * self.halo,
                            stride=self.window, padding=self.halo)
        return unfolded.reshape(batch, channels, count, -1).permute(0, 3, 2, 1).reshape(-1, count, channels)

    def _heads(self, tokens):
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _probabilities(logits, valid):
        weights = logits.float().masked_fill(~valid[:, None, None, :], -1e4).softmax(-1)
        weights = weights * valid[:, None, None, :].float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, source_p3, fused_p3, semantic_p4, return_aux=False):
        if (source_p3.ndim != 4 or source_p3.shape != fused_p3.shape
                or source_p3.shape[1] != self.channels or min(source_p3.shape[-2:]) < 1):
            raise ValueError("SDRF requires equal-shaped source/fused P3 BCHW features")
        if (semantic_p4.ndim != 4 or semantic_p4.shape[:2] != source_p3.shape[:2]
                or min(semantic_p4.shape[-2:]) < 1):
            raise ValueError("SDRF P4 batch/channels must match projected P3")
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and not return_aux:
            # Recompute only the new, stateless GN/LN branch. Baseline BN/FPN
            # is outside checkpoint, so its running statistics update once.
            return checkpoint(self._forward_impl, source_p3, fused_p3, semantic_p4, use_reentrant=False)
        return self._forward_impl(source_p3, fused_p3, semantic_p4, return_aux)

    def _forward_impl(self, source_p3, fused_p3, semantic_p4, return_aux=False):
        batch, _, height, width = source_p3.shape
        detail = self.detail_projection(source_p3)
        semantic = self.semantic_projection(semantic_p4)
        # Spatial correspondence is only bilinear resampling, NOT learned
        # anatomical alignment. No coarse semantic features enter V.
        with torch.autocast(device_type=source_p3.device.type, enabled=False):
            semantic = F.interpolate(semantic.float(), size=(height, width), mode="bilinear", align_corners=False)
        semantic = semantic.to(detail.dtype)
        pad_h, pad_w = (-height) % self.window, (-width) % self.window
        qkv = F.pad(self.detail_qkv(detail), (0, pad_w, 0, pad_h))
        qf, kf, vf = qkv.chunk(3, 1)
        qs = F.pad(self.semantic_query(semantic), (0, pad_w, 0, pad_h))
        mask = F.pad(torch.ones(batch, 1, height, width, device=detail.device), (0, pad_w, 0, pad_h))
        valid_keys = self._key_blocks(mask).squeeze(-1).bool()
        valid_queries = self._query_blocks(mask).squeeze(-1).bool()
        qf, qs = self._heads(self._query_blocks(qf)), self._heads(self._query_blocks(qs))
        kf, vf = self._heads(self._key_blocks(kf)), self._heads(self._key_blocks(vf))
        with torch.autocast(device_type=source_p3.device.type, enabled=False):
            appearance_logits = (qf.float() @ kf.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            appearance_logits = appearance_logits + self.relative_bias.float()[:, self.relative_index].unsqueeze(0)
            semantic_logits = (qs.float() @ kf.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            strength = self.semantic_strength_logit.float().sigmoid()[None, :, None, None]
            # Even ablations keep the graph, avoiding unused DDP parameters.
            guided_logits = appearance_logits + strength * semantic_logits * int(self.semantic_condition)
            appearance = self._probabilities(appearance_logits, valid_keys)
            guided = self._probabilities(guided_logits, valid_keys)
            differential = guided - appearance
            route = differential if self.differential else guided
            route = route * valid_queries[:, None, :, None].float()
            message = route @ vf.float()
        message = message.to(detail.dtype).transpose(1, 2).reshape(-1, self.window ** 2, self.dim)
        rows, cols = (height + pad_h) // self.window, (width + pad_w) // self.window
        relation = message.reshape(batch, rows, cols, self.window, self.window, self.dim).permute(
            0, 5, 1, 3, 2, 4).reshape(batch, self.dim, height + pad_h, width + pad_w)[:, :, :height, :width]
        normalized = self.output_norm(relation.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        delta = self.fuse(normalized).to(fused_p3.dtype)
        result = fused_p3 + self.residual_scale().to(fused_p3.dtype) * delta
        if return_aux:
            return result, {"appearance_attention": appearance, "guided_attention": guided,
                "differential_attention": differential, "valid_keys": valid_keys,
                "valid_queries": valid_queries, "relation_message": relation,
                "semantic_strength": strength.flatten(), "residual_scale": self.residual_scale()}
        return result
