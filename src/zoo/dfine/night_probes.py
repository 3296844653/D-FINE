"""Six independent SCB-S structural screening probes (not proven innovations).

All residual outputs start at zero. No GT-conditioned crops, query count change,
new losses, or inference-only transformations are used. Predicted sampling boxes
are detached, but the existing classification loss still trains shared features.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


NIGHT_PROBES = (
    "backbone_bandpass", "encoder_cross_gate", "query_local_init",
    "decoder_l2_local", "rw_adaptive_roi", "query_content_score",
)


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def project(channels, dim):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(channels, dim, 1, bias=False)),
        ("norm", nn.GroupNorm(8, dim)), ("act", nn.SiLU()),
    ]))


class BackboneBandPass(nn.Module):
    """P3 output-interface probe; P4/P5 backbone computations are unchanged."""

    def __init__(self, channels, dim=64):
        super().__init__()
        self.projection = project(channels, dim)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.wide = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.output = nn.Conv2d(4 * dim, channels, 1)
        zero_output(self.output)

    def forward(self, feature):
        detail = self.projection(feature)
        band = detail - F.avg_pool2d(detail, 3, stride=1, padding=1)
        residual = self.output(torch.cat((detail, self.local(detail), self.wide(detail), band), 1))
        return feature + 0.1 * residual


class EncoderCrossGate(nn.Module):
    """Restore local P3 contrast guided by P4, after ordinary FPN/PAN fusion."""

    def __init__(self, channels, dim=64):
        super().__init__()
        self.p3_projection = project(channels, dim)
        self.p4_projection = project(channels, dim)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.gate = nn.Conv2d(2 * dim, dim, 1)
        self.output = nn.Conv2d(2 * dim, channels, 1)
        zero_output(self.output)

    def forward(self, p3, p4):
        fine = self.p3_projection(p3)
        context = F.interpolate(self.p4_projection(p4), size=p3.shape[-2:],
                                mode="bilinear", align_corners=False)
        detail = self.local(fine - F.avg_pool2d(fine, 3, stride=1, padding=1))
        gate = self.gate(torch.cat((fine, context), 1)).sigmoid()
        return p3 + 0.1 * self.output(torch.cat((gate * detail, context), 1))


class AdaptiveROITokens(nn.Module):
    """Query-conditioned nine locations, bounded inside each predicted box.

    Uses at most 18 projected tokens/query, not a per-query high-channel map or
    Q-by-Q attention. Locations are not supervised/identified as hands or books.
    """

    def __init__(self, hidden_dim=256, dim=64, p2_channels=None):
        super().__init__()
        self.hidden_dim, self.dim = hidden_dim, dim
        self.p3_projection = project(hidden_dim, dim)
        self.p2_projection = project(p2_channels, dim) if p2_channels is not None else None
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.offset = nn.Linear(hidden_dim, 18)
        zero_output(self.offset)
        self.query_projection = nn.Linear(hidden_dim, dim)
        self.position = nn.Linear(2, dim, bias=False)
        self.scale_embedding = nn.Parameter(torch.zeros(2 if p2_channels else 1, dim))
        self.token_norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        axis = (torch.arange(3, dtype=torch.float32) + .5) / 3 - .5
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("relative_grid", torch.stack((xx, yy), -1).flatten(0, 1),
                             persistent=False)

    def relative_locations(self, query):
        offsets = .15 * self.offset(self.query_norm(query)).float().tanh()
        return (self.relative_grid + offsets.reshape(*query.shape[:2], 9, 2)).clamp(-.49, .49)

    def _sample(self, feature, boxes, relative):
        # FP32 coordinate arithmetic and sampling even under CUDA FP16 autocast.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            box = boxes.detach().float()
            xy = box[..., None, :2] + box[..., None, 2:] * relative.float()
            sampled = F.grid_sample(feature.float(), 2 * xy - 1, align_corners=False,
                                    mode="bilinear", padding_mode="zeros")
        return sampled.permute(0, 2, 3, 1).to(feature.dtype)

    def forward(self, query, boxes, p3, p2=None):
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError("Night ROI query must be [B,Q,hidden_dim]")
        if boxes.shape != (*query.shape[:2], 4):
            raise ValueError("Night ROI boxes must align with queries")
        if (p3.ndim != 4 or p3.shape[:2] != (query.shape[0], self.hidden_dim)):
            raise ValueError("Night ROI requires the projected encoder P3")
        if self.p2_projection is not None:
            if p2 is None or p2.ndim != 4 or p2.shape[:2] != (
                    query.shape[0], self.p2_projection[0].in_channels):
                raise ValueError("Night adaptive RW ROI requires backbone P2")
        relative = self.relative_locations(query)
        position = self.position(relative).to(query.dtype)
        tokens = [self._sample(self.p3_projection(p3), boxes, relative) + position + self.scale_embedding[0]]
        if self.p2_projection is not None:
            tokens.append(self._sample(self.p2_projection(p2), boxes, relative) + position + self.scale_embedding[1])
        tokens = self.token_norm(torch.cat(tokens, 2))
        content = self.query_projection(self.query_norm(query))
        logits = (self.key(tokens) * content.unsqueeze(2)).sum(-1).float() / math.sqrt(self.dim)
        weights = logits.softmax(2).to(tokens.dtype)
        local = (weights.unsqueeze(-1) * self.value(tokens)).sum(2)
        return content, local


class BoxLocalQueryAdapter(nn.Module):
    """A residual query update used independently at initialization OR layer 2."""

    def __init__(self, hidden_dim=256, dim=64):
        super().__init__()
        self.sampler = AdaptiveROITokens(hidden_dim, dim)
        self.fuse = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                  nn.SiLU(), nn.Linear(dim, hidden_dim))
        zero_output(self.fuse[-1])

    def forward(self, query, boxes, p3):
        content, local = self.sampler(query, boxes, p3)
        return query + .1 * self.fuse(torch.cat((content, local), -1))


class RWAdaptiveROI(nn.Module):
    """Opposite read/write logit corrections from adaptively sampled P2/P3."""

    def __init__(self, hidden_dim=256, p2_channels=64, dim=64):
        super().__init__()
        self.sampler = AdaptiveROITokens(hidden_dim, dim, p2_channels)
        self.contrast = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                      nn.SiLU(), nn.Linear(dim, 1))
        zero_output(self.contrast[-1])

    def forward(self, query, boxes, p3, p2):
        content, local = self.sampler(query, boxes, p3, p2)
        delta = .5 * self.contrast(torch.cat((content, local), -1)).tanh()
        # category_id 0=hand-raising, 1=read, 2=write. Hand logit is unchanged.
        return torch.cat((torch.zeros_like(delta), -delta, delta), -1)


class QueryContentScore(nn.Module):
    """Common logit correction: query + local appearance + box-bin certainty.

    Supervised ONLY by original VFL. This is not a calibrated probability of
    validity; it tests content-aware score ordering, without changing argmax.
    """

    def __init__(self, hidden_dim=256, dim=64):
        super().__init__()
        self.sampler = AdaptiveROITokens(hidden_dim, dim)
        self.score = nn.Sequential(nn.LayerNorm(2 * dim + 8),
                                   nn.Linear(2 * dim + 8, dim), nn.SiLU(), nn.Linear(dim, 1))
        zero_output(self.score[-1])

    def forward(self, query, boxes, p3, corners):
        content, local = self.sampler(query, boxes, p3)
        with torch.autocast(device_type=query.device.type, enabled=False):
            distribution = corners.detach().float().reshape(*query.shape[:2], 4, -1).softmax(-1)
            peak = distribution.amax(-1)
            entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1)
            entropy = entropy / math.log(distribution.shape[-1])
            quality = torch.cat((peak, entropy), -1).to(content.dtype)
        return .5 * self.score(torch.cat((content, local, quality), -1)).tanh()
