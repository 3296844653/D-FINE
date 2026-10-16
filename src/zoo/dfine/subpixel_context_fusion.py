"""SubpixelContextFusion: P2 phase evidence + P3 neighborhood relations.

Experimental feature mechanism, not proven anatomy, novelty or accuracy gain.
Four P2 subcells are retained rather than interpolated/averaged before selection.
P3/P4 condition fine-detail selection; the selected evidence conditions a bounded
P3 neighborhood read. Null values let both reads reject unhelpful evidence.
The residual enters FPN P3 BEFORE PAN, not a query or final class-score branch.
No boxes, confidence thresholds, class axes, GT labels or extra losses are used.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def projection(source, dim):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(source, dim, 1, bias=False)),
        ("norm", nn.GroupNorm(8, dim)),
        ("act", nn.SiLU()),
    ]))


class SubpixelContextFusion(nn.Module):
    MAX_SCALE = .5

    def __init__(self, channels=256, p2_channels=64, dim=32, dilation=2,
                 residual_init=.1, use_checkpoint=True,
                 phase_selection=True, context_relations=True):
        super().__init__()
        if any(type(v) is not int for v in (channels, p2_channels, dim, dilation)):
            raise ValueError("SCF dimensions/dilation must be integers")
        if min(channels, p2_channels) < 16 or dim < 16 or dim % 8 or dilation not in (1, 2, 3):
            raise ValueError("SCF needs channels>=16, dim>=16 divisible by 8, dilation=1/2/3")
        if type(residual_init) not in (int, float) or not math.isfinite(residual_init) or not 0 < residual_init < self.MAX_SCALE:
            raise ValueError("SCF residual_init must be finite in (0,0.5)")
        if any(type(v) is not bool for v in (use_checkpoint, phase_selection, context_relations)):
            raise ValueError("SCF controls must be boolean")
        self.channels, self.p2_channels, self.dim = channels, p2_channels, dim
        self.dilation, self.use_checkpoint = dilation, use_checkpoint
        self.phase_selection, self.context_relations = phase_selection, context_relations
        self.enabled = True  # Runtime ON/OFF: identical construction/checkpoint schema.
        self.register_buffer("settings", torch.tensor([
            1, channels, p2_channels, dim, dilation, int(use_checkpoint),
            int(phase_selection), int(context_relations)], dtype=torch.int64))
        self.fine_projection = projection(p2_channels, dim)
        self.query_projection = projection(channels, dim)
        self.semantic_projection = projection(channels, dim)
        self.context_projection = projection(channels, dim)
        self.phase_key = nn.Conv2d(dim, dim, 1, bias=False)
        self.phase_position = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.phase_position, std=.02)
        self.phase_bias = nn.Parameter(torch.zeros(4))
        self.phase_null_bias = nn.Parameter(torch.zeros(()))
        self.evidence_query = nn.Conv2d(dim, dim, 1, bias=False)
        self.context_key = nn.Conv2d(dim, dim, 1, bias=False)
        self.context_value = nn.Conv2d(dim, dim, 1, bias=False)
        self.context_bias = nn.Parameter(torch.zeros(9))
        self.context_null_bias = nn.Parameter(torch.zeros(()))
        self.gates = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.fuse = projection(4 * dim, dim)
        self.output = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.raw_scale = nn.Parameter(torch.tensor(math.atanh(residual_init / self.MAX_SCALE)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        settings = state_dict.get(prefix + "settings")
        if settings is not None and not torch.equal(settings.cpu(), self.settings.cpu()):
            error_msgs.append(f"{prefix}SCF checkpoint/config settings disagree")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def residual_scale(self):
        return self.MAX_SCALE * self.raw_scale.tanh()

    @staticmethod
    def subcells(feature, height, width):
        """Keep TL/TR/BL/BR in that order; pad only missing bottom/right cells.

        HGNet ceil-stride grids permit P2 spatial size 2*P3 or 2*P3-1.
        Reject other geometry rather than silently interpolate phase evidence.
        """
        b, c, h, w = feature.shape
        if h not in (2 * height - 1, 2 * height) or w not in (2 * width - 1, 2 * width):
            raise ValueError("SCF requires aligned stride-4 P2 / stride-8 P3 grids")
        pad = (0, 2 * width - w, 0, 2 * height - h)
        packed = F.pixel_unshuffle(F.pad(feature, pad), 2)
        cells = packed.reshape(b, c, 4, height, width).permute(0, 2, 1, 3, 4)
        valid = F.pixel_unshuffle(F.pad(feature.new_ones(1, 1, h, w), pad), 2).bool()
        return cells, valid

    def neighbors(self, feature):
        b, c, h, w = feature.shape
        return F.unfold(feature, 3, dilation=self.dilation, padding=self.dilation).reshape(
            b, c, 9, h, w).permute(0, 2, 1, 3, 4)

    @staticmethod
    def probabilities(logits, valid, null_bias, selection=True):
        # FP32 softmax, explicit padding mask, and zero-valued null candidate.
        logits = logits.float()
        null = null_bias.float().expand(logits.shape[0], 1, *logits.shape[-2:])
        if not selection:
            logits, null = logits * 0, null * 0
        valid = valid.expand(logits.shape[0], -1, -1, -1)
        logits = logits.masked_fill(~valid, -torch.inf)
        return torch.cat((logits, null), 1).softmax(1)

    def forward(self, p2, p3, p4, return_aux=False):
        if p2.ndim != 4 or p3.ndim != 4 or p4.ndim != 4:
            raise ValueError("SCF inputs must be BCHW")
        if (p2.shape[:2] != (p3.shape[0], self.p2_channels)
                or p3.shape[1] != self.channels or p4.shape[:2] != p3.shape[:2]
                or min(*p2.shape[-2:], *p3.shape[-2:], *p4.shape[-2:]) < 1
                or any(a > b for a,b in zip(p4.shape[-2:],p3.shape[-2:]))):
            raise ValueError("SCF input batch/channels/spatial sizes disagree")
        # Native backbone P2 may be FP32 while Encoder P3 is autocast FP16.
        # Cast dtype in the new branch, never silently move input devices.
        if any(f.device != p3.device for f in (p2, p4)):
            raise ValueError("SCF input devices disagree")
        if not self.enabled:
            return (p3, {}) if return_aux else p3
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and not return_aux:
            # Only stateless new GN operations recompute. Original FPN/PAN/BN
            # are outside this checkpoint and update their statistics once.
            return checkpoint(self._forward_impl, p2, p3, p4, use_reentrant=False)
        return self._forward_impl(p2, p3, p4, return_aux)

    def _forward_impl(self, p2, p3, p4, return_aux=False):
        h, w = p3.shape[-2:]
        fine = self.fine_projection(p2.to(p3.dtype))
        query = self.query_projection(p3)
        semantic = self.semantic_projection(p4.to(p3.dtype))
        semantic = F.interpolate(semantic, size=(h, w), mode="bilinear", align_corners=False)
        cells, valid = self.subcells(fine, h, w)
        keys, _ = self.subcells(self.phase_key(fine), h, w)
        keys = keys + self.phase_position.to(keys.dtype)[None, :, :, None, None]
        logits = ((query + semantic).float()[:, None] * keys.float()).sum(2) / math.sqrt(self.dim)
        weights = self.probabilities(logits + self.phase_bias[None, :, None, None],
            valid, self.phase_null_bias, self.phase_selection)
        real = weights[:, :4].to(query.dtype)
        selected = (real[:, :, None] * cells).sum(1)
        mean = (cells * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        phase_mass = real.sum(1, keepdim=True)
        phase_contrast = selected - phase_mass * mean

        context = self.context_projection(p3)
        context_query = query + semantic + self.evidence_query(selected)
        context_keys = self.neighbors(self.context_key(context))
        values = self.neighbors(self.context_value(context))
        neighbor_valid = self.neighbors(p3.new_ones(1, 1, h, w))[:, :, 0].bool()
        context_logits = (context_query.float()[:, None] * context_keys.float()).sum(2) / math.sqrt(self.dim)
        context_weights = self.probabilities(context_logits + self.context_bias[None, :, None, None],
            neighbor_valid, self.context_null_bias)
        context_real = context_weights[:, :9].to(query.dtype)
        # Read relative evidence, not arbitrary whole-cell appearance addition.
        relation = (context_real[:, :, None] * (values - values[:, 4:5])).sum(1)
        if not self.context_relations:
            relation = relation * 0  # Same graph/parameters; structural ablation.
        gates = self.gates(torch.cat((query, semantic, selected.abs(), relation.abs()), 1)).sigmoid()
        detail_gate, relation_gate = gates.chunk(2, 1)
        detail, relation = selected * detail_gate, relation * relation_gate
        fused = torch.cat((detail, phase_contrast, relation, detail * relation), 1)
        availability = .5 * (phase_mass + context_real.sum(1, keepdim=True))
        correction = availability * self.output(self.fuse(fused))
        scale = self.residual_scale().to(p3.dtype)
        result = p3 + scale * correction.to(p3.dtype)
        if return_aux:
            return result, dict(phase_weights=weights, phase_valid=valid,
                context_weights=context_weights, context_valid=neighbor_valid,
                phase_contrast=phase_contrast, detail=detail, relation=relation,
                residual=scale * correction, availability=availability)
        return result
